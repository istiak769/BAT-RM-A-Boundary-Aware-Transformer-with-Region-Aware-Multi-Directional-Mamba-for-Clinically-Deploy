
"""
BAT-RM U-Net: Boundary-Aware Transformer and Region-Mamba U-Net
for OAR Segmentation in Cervical Cancer Radiotherapy

Key fixes and enhancements over the original draft:
  1.  GatedBATBlock: replaced O(N^2) full spatial self-attention with
      efficient strip-pooling multi-head attention — reduces VRAM from
      ~268 M to O(N) while preserving boundary-aware gating.
  2.  B_hat boundary prediction head added inside GatedBATBlock so that
      the BRAF module and boundary BCE loss receive the correct signal.
  3.  RegionMambaBlock: proper 4-direction (N/S/E/W) sequential scanning
      with independent GRU-based SSM streams, summed as in the paper.
  4.  BRAFModule: corrected to use B_hat (not the Sobel gate) for
      boundary-guided refinement, matching the paper exactly.
  5.  BAT_RM_UNet.forward: BRAF output injected at the correct 128×128
      decoder level (Stage 4), replacing the E3 skip connection.
  6.  Model now returns (seg_logits, b_hat) so that the loss function
      can supervise the boundary head directly.
  7.  Channel widths kept at the half-scale (32/64/128/256/512) used in
      the original code — all internal dims updated consistently.
  8.  Encoder uses a residual skip inside each block for better gradient
      flow (lightweight 1×1 projection when in_ch ≠ out_ch).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: Encoder Block with optional residual connection
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """
    Two-layer 3×3 conv block with BN + ReLU.
    Adds a residual (identity / 1×1-projected) connection for better
    gradient flow through the deep encoder.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Residual projection only when channel dim changes
        self.residual = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.residual(x)


# ---------------------------------------------------------------------------
# Gated Boundary-Aware Transformer (BAT) Block  [FIX 1, 2, 3]
# ---------------------------------------------------------------------------

class GatedBATBlock(nn.Module):
    """
    Sobel-Gated Multi-Head Self-Attention for boundary-aware feature
    refinement (Section 3.3 of the paper).

    Efficient strip-pooling attention
    ---------------------------------
    Full spatial self-attention at 128×128 produces an N×N = 16 384×16 384
    attention matrix that requires ~4 GB for a single sample — impractical.

    We use *strip-pooling*: the feature map is pooled along each axis into
    H or W representative tokens, forming two 1-D attention problems of
    size H and W (both ≤ 128).  This is consistent with the paper's goal of
    capturing long-range interactions along organ perimeters at linear cost.

    The Sobel gate G ∈ [0,1]^{H×W} is applied to Q and K before attention,
    exactly as described in the paper (Equations 6-8).

    Returns
    -------
    out  : refined feature map  (B, C, H, W)
    b_hat: predicted boundary map (B, 1, H, W)  — supervised with BCE loss
    """

    def __init__(self, in_ch: int, num_heads: int = 8):
        super().__init__()
        assert in_ch % num_heads == 0, "in_ch must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = in_ch // num_heads

        # Fixed Sobel kernels (not trainable)
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # 1×1 conv to collapse channel dim of gradient map → scalar gate
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Q / K / V projections for strip-pooling attention
        self.q_proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)

        # Learnable residual scale (γ), initialised to 0
        self.gamma = nn.Parameter(torch.zeros(1))

        # Boundary prediction head: B_hat ∈ [0,1]^{H×W}
        # Two-layer MLP (implemented as 1×1 convs) + sigmoid, as in paper
        self.boundary_head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_ch // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm  # used as post-attention norm (applied in-place)
        self.post_norm = nn.GroupNorm(num_groups=8, num_channels=in_ch)

    # ------------------------------------------------------------------
    # Internal: Sobel gradient gate on multi-channel features
    # ------------------------------------------------------------------
    def _compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-pixel boundary gate G ∈ [0,1]^{B,1,H,W}.
        Sobel is applied channel-by-channel and the gradient magnitudes
        are summed before the sigmoid gate projection.
        """
        B, C, H, W = x.shape
        # Reshape so each channel is treated as a 1-channel image
        x_flat = x.reshape(B * C, 1, H, W)
        gx = F.conv2d(x_flat, self.sobel_x, padding=1)
        gy = F.conv2d(x_flat, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)           # (B*C,1,H,W)
        grad_mag = grad_mag.reshape(B, C, H, W)                    # (B,C,H,W)
        gate = self.gate_conv(grad_mag)                            # (B,1,H,W)
        return gate  # already sigmoid-activated

    # ------------------------------------------------------------------
    # Internal: strip-pooling self-attention along one axis
    # ------------------------------------------------------------------
    def _strip_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, axis: int
    ) -> torch.Tensor:
        """
        Efficient 1-D attention pooled along 'axis' (2=height, 3=width).
        Q, K, V: (B, C, H, W)
        Returns attended feature map (B, C, H, W).
        """
        B, C, H, W = Q.shape
        other = W if axis == 2 else H   # the axis we pool over

        # Pool along the *other* axis to get H (or W) tokens of size C
        q_pool = Q.mean(dim=axis)       # (B, C, W) or (B, C, H)
        k_pool = K.mean(dim=axis)
        v_pool = V.mean(dim=axis)

        # Reshape for multi-head: (B, heads, tokens, d_k)
        q_pool = q_pool.permute(0, 2, 1).reshape(B, other, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k_pool = k_pool.permute(0, 2, 1).reshape(B, other, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v_pool = v_pool.permute(0, 2, 1).reshape(B, other, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        attn = torch.matmul(q_pool, k_pool.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out_pool = torch.matmul(attn, v_pool)                          # (B, heads, tokens, d_k)
        out_pool = out_pool.permute(0, 2, 1, 3).reshape(B, other, C)  # (B, tokens, C)
        out_pool = out_pool.permute(0, 2, 1)                           # (B, C, tokens)

        # Broadcast back to (B, C, H, W)
        if axis == 2:   # pooled along H → tokens are W-indexed → expand H
            out = out_pool.unsqueeze(2).expand(-1, -1, H, -1)
        else:           # pooled along W → tokens are H-indexed → expand W
            out = out_pool.unsqueeze(3).expand(-1, -1, -1, W)
        return out

    def forward(self, x: torch.Tensor):
        # ---- Sobel gate ----
        gate = self._compute_gate(x)           # (B, 1, H, W)

        # ---- Project and gate Q, K (paper Eq. 6) ----
        Q = self.q_proj(x) * gate             # gate broadcast over C
        K = self.k_proj(x) * gate
        V = self.v_proj(x)

        # ---- Strip-pooling attention (H and W axes) ----
        attn_h = self._strip_attention(Q, K, V, axis=3)   # pool over W
        attn_w = self._strip_attention(Q, K, V, axis=2)   # pool over H
        attn_out = self.out_proj(attn_h + attn_w)

        # ---- Residual + post-norm ----
        out = self.post_norm(self.gamma * attn_out + x)

        # ---- Boundary prediction head (B_hat) ----
        b_hat = self.boundary_head(out)        # (B, 1, H, W), values in [0,1]

        return out, b_hat, gate


# ---------------------------------------------------------------------------
#  (RM) Block "Multi-Directional Recurrent Context Module"
# ---------------------------------------------------------------------------

class RegionMambaBlock(nn.Module):
    """
    Multi-directional SSM block operating at the bottleneck (E5, 32×32).

    True Mamba requires the `mamba-ssm` package.  We implement a faithful
    proxy using bidirectional GRUs — this matches the *inductive bias* of
    SSMs (sequential, recurrent, linear-time) and avoids the quadratic cost
    of transformers, making it a valid drop-in for ablation and publication.

    Four cardinal directions are scanned independently and their outputs
    are summed, as in the paper (Eq. 12).

    If you install `mamba-ssm`, swap the GRU lines for:
        from mamba_ssm import Mamba
        self.ssm_N = Mamba(d_model=in_ch, d_state=16, d_conv=4, expand=2)
        ... (one per direction)
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.in_ch = in_ch
        self.norm = nn.LayerNorm(in_ch)

        # One GRU per direction; bidirectional=False to match unidirectional scan
        self.ssm_N = nn.GRU(in_ch, in_ch, batch_first=True)   # North→South
        self.ssm_S = nn.GRU(in_ch, in_ch, batch_first=True)   # South→North
        self.ssm_E = nn.GRU(in_ch, in_ch, batch_first=True)   # East→West (row-major L→R)
        self.ssm_W = nn.GRU(in_ch, in_ch, batch_first=True)   # West→East (row-major R→L)

        # SiLU gate before selective scan (paper Eq. 11)
        self.in_proj = nn.Linear(in_ch, in_ch)
        self.gate_proj = nn.Linear(in_ch, in_ch)

        self.out_proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.post_norm = nn.GroupNorm(num_groups=8, num_channels=in_ch)

    def _scan(self, ssm: nn.GRU, tokens: torch.Tensor) -> torch.Tensor:
        """Run SSM on token sequence; apply SiLU gate."""
        gate = F.silu(self.gate_proj(tokens))
        out, _ = ssm(self.in_proj(tokens))
        return out * gate   # selective scan via multiplicative gating

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        shortcut = x

        # Normalise over channel dim
        x_norm = self.norm(x.permute(0, 2, 3, 1).reshape(B, H * W, C))

        # ---- North→South: column-major top→bottom ----
        # Reshape to (B, W, H, C) then flatten inner → (B*W, H, C)
        ns_tokens = x_norm.reshape(B, H, W, C).permute(0, 2, 1, 3).reshape(B * W, H, C)
        ns_out = self._scan(self.ssm_N, ns_tokens)                   # (B*W, H, C)
        ns_out = ns_out.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H * W, C)

        # ---- South→North: column-major bottom→top ----
        sn_tokens = ns_tokens.flip(dims=[1])
        sn_out = self._scan(self.ssm_S, sn_tokens).flip(dims=[1])    # (B*W, H, C)
        sn_out = sn_out.reshape(B, W, H, C).permute(0, 2, 1, 3).reshape(B, H * W, C)

        # ---- East→West: row-major left→right ----
        ew_tokens = x_norm.reshape(B * H, W, C)
        ew_out = self._scan(self.ssm_E, ew_tokens)                   # (B*H, W, C)
        ew_out = ew_out.reshape(B, H * W, C)

        # ---- West→East: row-major right→left ----
        we_tokens = ew_tokens.flip(dims=[1])
        we_out = self._scan(self.ssm_W, we_tokens).flip(dims=[1])    # (B*H, W, C)
        we_out = we_out.reshape(B, H * W, C)

        # ---- Aggregate (paper Eq. 12: sum over directions) ----
        f_rm = (ns_out + sn_out + ew_out + we_out)                   # (B, H*W, C)
        f_rm = f_rm.reshape(B, H, W, C).permute(0, 3, 1, 2)         # (B, C, H, W)

        out = self.post_norm(self.out_proj(f_rm) + shortcut)
        return out


# ---------------------------------------------------------------------------
# Boundary-Region Attention Fusion (BRAF) Module  [FIX 4, 5]
# ---------------------------------------------------------------------------

class BRAFModule(nn.Module):
    """
    Fuses BAT (boundary-aware, 128×128) and RM (region-aware, 32×32)
    features via learnable attention, then refines with the predicted
    boundary map B_hat (paper Section 3.5 / Equations 13-16).

    Parameters
    ----------
    bat_ch : int   channel width of F_BAT (128-channel features at E3)
    rm_ch  : int   channel width of F_RM  (512-channel features at E5)
    """

    def __init__(self, bat_ch: int = 128, rm_ch: int = 512):
        super().__init__()
        # Project RM channels down to match BAT channels
        self.rm_project = nn.Sequential(
            nn.Conv2d(rm_ch, bat_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(bat_ch),
            nn.ReLU(inplace=True),
        )

        # Spatial-channel attention gate α (paper Eq. 14)
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(bat_ch + bat_ch, bat_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bat_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(bat_ch, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Boundary-guided refinement projection (paper Eq. 16)
        self.boundary_modulate = nn.Sequential(
            nn.Conv2d(1, bat_ch, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Final conv after fusion
        self.refine = nn.Sequential(
            nn.Conv2d(bat_ch, bat_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bat_ch),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        f_bat: torch.Tensor,   # (B, bat_ch, 128, 128)
        f_rm: torch.Tensor,    # (B, rm_ch,  32,  32)
        b_hat: torch.Tensor,   # (B, 1,      128, 128)  ← boundary prediction
    ) -> torch.Tensor:

        # 1. Spatial alignment: upsample RM → BAT resolution
        f_rm_up = F.interpolate(f_rm, size=f_bat.shape[2:], mode="bilinear", align_corners=False)
        f_rm_aligned = self.rm_project(f_rm_up)   # (B, bat_ch, H_bat, W_bat)

        # 2. Learnable convex combination (paper Eq. 14-15)
        alpha = self.alpha_conv(torch.cat([f_bat, f_rm_aligned], dim=1))  # (B,1,H,W)
        f_inter = alpha * f_bat + (1.0 - alpha) * f_rm_aligned            # (B,bat_ch,H,W)

        # 3. Boundary-guided refinement using B_hat (paper Eq. 16)
        #    b_hat is already at (B,1,128,128) — same spatial size as f_inter
        boundary_weight = self.boundary_modulate(b_hat)                   # (B,bat_ch,H,W)
        f_fuse = f_inter * boundary_weight

        return self.refine(f_fuse)   # (B, bat_ch, 128, 128)


# ---------------------------------------------------------------------------
# BAT-RM U-Net  [FIX 6 — decoder injection, FIX return signature]
# ---------------------------------------------------------------------------

class BAT_RM_UNet(nn.Module):
    """
    Boundary-Aware Transformer and Region-Mamba U-Net (BAT-RM U-Net).

    Input  : (B, 3, 512, 512)   — 3-channel CT slice (e.g. windowed HU)
    Outputs: tuple(
        logits : (B, n_classes, 512, 512)  — segmentation logits
        b_hat  : (B, 1,         128, 128)  — boundary probability map
    )

    The b_hat output is used by boundary_bce_loss during training.
    At inference, only logits is needed.

    Encoder channel widths (half-scale vs. paper for memory efficiency):
        E1: 32,  E2: 64,  E3: 128 (→ BAT),  E4: 256,  E5: 512 (→ RM)

    Decoder injection:
        BRAF output (128-ch, 128×128) replaces the E3 skip at Stage 4,
        which operates at the 128×128 resolution — matching the BAT level.
    """

    def __init__(self, n_classes: int, in_channels: int = 3):
        super().__init__()

        # ---- Encoder ----
        self.pool = nn.MaxPool2d(2)
        self.e1 = EncoderBlock(in_channels, 32)    # → 512×512×32
        self.e2 = EncoderBlock(32, 64)             # → 256×256×64
        self.e3 = EncoderBlock(64, 128)            # → 128×128×128  (BAT input)
        self.e4 = EncoderBlock(128, 256)           # → 64×64×256
        self.e5 = EncoderBlock(256, 512)           # → 32×32×512    (RM input)

        # ---- Specialist branches ----
        self.bat = GatedBATBlock(in_ch=128, num_heads=8)
        self.rm  = RegionMambaBlock(in_ch=512)
        self.braf = BRAFModule(bat_ch=128, rm_ch=512)

        # ---- Decoder ----
        # Stage 5: 32×32 → 64×64  |  up(512) + skip E4(256) = 768
        self.up5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d5  = EncoderBlock(512, 256)

        # Stage 4: 64×64 → 128×128  |  up(256) + BRAF(128) = 384
        # BRAF replaces the raw E3 skip here (paper: BRAF output fed to decoder)
        self.up4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d4  = EncoderBlock(256, 128)   # 128 (up) + 128 (BRAF) = 256

        # Stage 3: 128×128 → 256×256  |  up(128) + skip E2(64) = 192
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d3  = EncoderBlock(128, 64)    # 64 (up) + 64 (E2) = 128

        # Stage 2: 256×256 → 512×512  |  up(64) + skip E1(32) = 96
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d2  = EncoderBlock(64, 32)     # 32 (up) + 32 (E1) = 64

        # Final 1×1 segmentation head
        self.out_conv = nn.Conv2d(32, n_classes, kernel_size=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]

        # ---- Encoder ----
        s1 = self.e1(x)                    # 512×512×32
        s2 = self.e2(self.pool(s1))        # 256×256×64
        s3 = self.e3(self.pool(s2))        # 128×128×128
        s4 = self.e4(self.pool(s3))        # 64×64×256
        s5 = self.e5(self.pool(s4))        # 32×32×512

        # ---- Specialist branches ----
        f_bat, b_hat, _gate = self.bat(s3)  # f_bat: 128×128×128, b_hat: 128×128×1
        f_rm  = self.rm(s5)                 # 32×32×512
        f_fuse = self.braf(f_bat, f_rm, b_hat)  # 128×128×128

        # ---- Decoder ----
        # Stage 5: bottleneck → 64×64
        x5 = self.up5(s5)
        x5 = self._match_and_cat(x5, s4)
        x5 = self.d5(x5)

        # Stage 4: 64×64 → 128×128  — use BRAF output instead of raw E3 skip
        x4 = self.up4(x5)
        x4 = self._match_and_cat(x4, f_fuse)   # f_fuse is already 128×128
        x4 = self.d4(x4)

        # Stage 3: 128×128 → 256×256
        x3 = self.up3(x4)
        x3 = self._match_and_cat(x3, s2)
        x3 = self.d3(x3)

        # Stage 2: 256×256 → 512×512
        x2 = self.up2(x3)
        x2 = self._match_and_cat(x2, s1)
        x2 = self.d2(x2)

        # Final output — restore to input resolution in case of odd sizes
        logits = self.out_conv(x2)
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        return logits, b_hat

    @staticmethod
    def _match_and_cat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Align spatial dims of 'a' to 'b', then concatenate along channel."""
        if a.shape[2:] != b.shape[2:]:
            a = F.interpolate(a, size=b.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([a, b], dim=1)


# ---------------------------------------------------------------------------
# Initialisation (keep original variable name for downstream compatibility)
# ---------------------------------------------------------------------------

# These are expected to be defined in the calling scope:
#   n_classes : int    (8 for the cervix OAR task)
#   DEVICE    : str    ('cuda' or 'cpu')

segmentation_model = BAT_RM_UNet(n_classes=n_classes, in_channels=3).to(DEVICE)

print(segmentation_model)
print(
    "Trainable parameters:",
    sum(p.numel() for p in segmentation_model.parameters() if p.requires_grad),
)





"""
Loss functions and metrics for BAT-RM U-Net OAR segmentation.
Class weights are computed dynamically in the training cell —
this cell only defines loss functions, metrics, and the optimizer.
"""

import torch
import torch.nn.functional as F
import numpy as np


########################################
# Utilities
########################################

def mask_to_one_hot(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer mask (B,H,W) to one-hot float (B,C,H,W)."""
    return (
        F.one_hot(mask, num_classes=num_classes)
        .permute(0, 3, 1, 2)
        .float()
    )


########################################
# Soft Dice Loss (class-balanced)
########################################

def dice_loss(
    y_true: torch.Tensor,          # (B, C, H, W) one-hot
    logits: torch.Tensor,          # (B, C, H, W) raw logits
    class_weights: torch.Tensor,   # (C,)
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)

    intersection   = torch.sum(y_true * probs, dim=(2, 3))   # (B, C)
    sum_true       = torch.sum(y_true,         dim=(2, 3))
    sum_pred       = torch.sum(probs,          dim=(2, 3))

    dice_per_class = (2.0 * intersection + smooth) / (sum_true + sum_pred + smooth)
    weighted       = (dice_per_class * class_weights).sum(dim=1) / class_weights.sum()
    return 1.0 - weighted.mean()


########################################
# Tversky Loss
# Use instead of Dice when FN cost matters more (small bowel, GTV)
# alpha=0.3 (FP penalty), beta=0.7 (FN penalty)
########################################

def tversky_loss(
    y_true: torch.Tensor,
    logits: torch.Tensor,
    class_weights: torch.Tensor,
    alpha: float = 0.3,
    beta: float  = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    tp    = torch.sum(y_true * probs,           dim=(2, 3))
    fp    = torch.sum((1 - y_true) * probs,     dim=(2, 3))
    fn    = torch.sum(y_true * (1 - probs),     dim=(2, 3))

    tversky  = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    weighted = (tversky * class_weights).sum(dim=1) / class_weights.sum()
    return 1.0 - weighted.mean()


########################################
# Multi-class Focal Loss
########################################

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,          # (B, H, W) integer labels
    class_weights: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce  = F.cross_entropy(logits, targets, weight=class_weights, reduction="none")
    pt  = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


########################################
# Boundary BCE Loss
########################################

def boundary_bce_loss(
    b_hat: torch.Tensor,   # (B, 1, H_bat, W_bat) — model boundary head output
    masks: torch.Tensor,   # (B, H, W)             — integer segmentation labels
) -> torch.Tensor:
    """
    Supervise the BAT boundary head with morphological ground-truth boundaries.
    Mask is downsampled to b_hat resolution (128x128) before computing boundaries.
    """
    H_bat, W_bat = b_hat.shape[2:]

    masks_small = F.interpolate(
        masks.unsqueeze(1).float(), size=(H_bat, W_bat), mode="nearest"
    )

    dilate      = F.max_pool2d(masks_small,   kernel_size=3, stride=1, padding=1)
    erode       = -F.max_pool2d(-masks_small, kernel_size=3, stride=1, padding=1)
    gt_boundary = (dilate - erode).clamp(0, 1)   # (B, 1, H_bat, W_bat)

    return F.binary_cross_entropy(b_hat, gt_boundary)


########################################
# Region Smoothness Regularisation
########################################

def region_smoothness_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    y_hat  = torch.softmax(logits, dim=1)
    y_true = mask_to_one_hot(masks, n_classes).to(logits.device)

    def spatial_gradients(t):
        return t[:, :, :, 1:] - t[:, :, :, :-1], t[:, :, 1:, :] - t[:, :, :-1, :]

    hat_gx,  hat_gy  = spatial_gradients(y_hat)
    true_gx, true_gy = spatial_gradients(y_true)

    return F.mse_loss(hat_gx, true_gx) + F.mse_loss(hat_gy, true_gy)


########################################
# Combined Loss
########################################

def combined_loss(
    mask_indices: torch.Tensor,          # (B, H, W) integer labels
    logits: torch.Tensor,                # (B, C, H, W) segmentation logits
    b_hat: torch.Tensor,                 # (B, 1, H_bat, W_bat) boundary head output
    class_weights_tensor: torch.Tensor,  # (C,) — computed dynamically in train cell
    n_classes: int,
) -> torch.Tensor:
    """
    Loss weights:
        Dice         0.50  — primary overlap objective
        Focal        0.20  — hard-example mining for class imbalance
        Boundary BCE 0.20  — boundary head supervision
        Smoothness   0.10  — shape regularisation
    """
    y_true = mask_to_one_hot(mask_indices, num_classes=n_classes).to(logits.device)

    l_dice  = dice_loss(y_true, logits, class_weights_tensor)
    l_focal = focal_loss(logits, mask_indices, class_weights_tensor)
    l_bd    = boundary_bce_loss(b_hat, mask_indices)
    l_reg   = region_smoothness_loss(logits, mask_indices, n_classes)

    return 0.50 * l_dice + 0.20 * l_focal + 0.20 * l_bd + 0.10 * l_reg


########################################
# Metrics
########################################

def pixel_accuracy(logits: torch.Tensor, mask_indices: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == mask_indices).float().mean().item()


def mean_iou_score(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    ious = []
    for cls in range(num_classes):
        inter = ((preds == cls) & (targets == cls)).sum().item()
        union = ((preds == cls) | (targets == cls)).sum().item()
        ious.append(1.0 if union == 0 else inter / union)
    return float(np.mean(ious))


def dice_coefficient(
    y_true: torch.Tensor,
    logits: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs        = torch.softmax(logits, dim=1)
    y_true       = y_true.float()
    intersection = torch.sum(y_true * probs, dim=(1, 2, 3))
    sum_true     = torch.sum(y_true,         dim=(1, 2, 3))
    sum_pred     = torch.sum(probs,          dim=(1, 2, 3))
    dice         = (2.0 * intersection + smooth) / (sum_true + sum_pred + smooth)
    return dice.mean()


########################################
# Optimizer — AdamW + cosine annealing scheduler
#
# Why AdamW over Adam:
#   Adam absorbs weight decay into the adaptive gradient scaling — so
#   larger weights are not actually penalised proportionally.  AdamW
#   fixes this with decoupled weight decay applied directly to parameters,
#   giving cleaner L2 regularisation.  For boundary/fusion modules where
#   precise weight magnitudes affect edge sharpness, this matters.
#
# LR reduced to 1e-4 (from 1e-3):
#   AdamW converges more reliably at a lower base LR.  The scheduler
#   handles warm restarts so the effective LR still varies across training.
########################################

LEARNING_RATE = 1e-4

optimizer = torch.optim.AdamW(
    segmentation_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5,    # paper value; keeps boundary/fusion weights in check
    betas=(0.9, 0.999),
    eps=1e-8,
)

# Cosine annealing with warm restarts (paper Section 3.6)
#   T_0=50   : first restart after 50 epochs
#   T_mult=2 : each subsequent cycle doubles in length (50 → 100 → 200)
#   eta_min  : LR floor so the model still learns slowly in late cycles
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=50,
    T_mult=2,
    eta_min=1e-6,
)

print(segmentation_model)
print(
    "Trainable parameters:",
    sum(p.numel() for p in segmentation_model.parameters() if p.requires_grad),
)
