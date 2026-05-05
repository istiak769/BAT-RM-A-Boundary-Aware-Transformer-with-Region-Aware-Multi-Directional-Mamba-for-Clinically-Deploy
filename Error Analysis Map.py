# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:00:08 2026

@author: Hulu
"""


import os
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import normalize
import tensorflow as tf
from keras.saving import register_keras_serializable
from PIL import Image

# ===============================
# 1. CUSTOM OBJECTS (same as original)
# ===============================
@register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    loss = tf.pow(1 - y_pred, gamma) * cross_entropy
    return tf.reduce_mean(loss, axis=-1)

@register_keras_serializable()
def soft_dice_loss(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    sum_true = tf.reduce_sum(y_true, axis=(1, 2, 3))
    sum_pred = tf.reduce_sum(y_pred, axis=(1, 2, 3))
    dice_coefficient = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
    return tf.reduce_mean(1 - dice_coefficient)

@register_keras_serializable()
def combined_loss(y_true, y_pred, gamma=2.0, alpha=0.5):
    return alpha * focal_loss(y_true, y_pred, gamma) + (1 - alpha) * soft_dice_loss(y_true, y_pred)

@register_keras_serializable()
class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name='mean_iou', dtype='float32',
                 ignore_class=None, sparse_y_true=True, sparse_y_pred=True, axis=-1):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype, ignore_class=ignore_class)
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({'sparse_y_true': self.sparse_y_true,
                       'sparse_y_pred': self.sparse_y_pred,
                       'axis': self.axis})
        return config

# ===============================
# 2. CONFIGURATION — EDIT THESE
# ===============================
MODEL_PATH   = r'I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/UNET_BM/200 Epochs/best_model.keras'
IMAGES_FOLDER = r'I:/Radiotherapy/Cervix/cervix_small_set/images'
MASKS_FOLDER  = r'I:/Radiotherapy/Cervix/cervix_small_set/masks'
SAVE_EXCEL    = r'I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/UNET_BM/200 Epochs/segmentation_metrics_detailed_epoch_200_generated.xlsx'
IMAGE_SIZE    = (512, 512)

# Class definitions — must match your training setup
class_names = {
    0: 'Background',
    1: 'BODY',
    2: 'URINARY BLADDER',
    3: 'SMALL BOWEL',
    4: 'RECTUM',
    5: 'FEMORAL HEAD',
    6: 'GTV',
    7: 'CTV',
}
n_classes = len(class_names)

# Mask RGB → class ID mapping (used to decode ground truth mask PNG)
# Each RGB tuple maps to a class ID
rgb_to_class = {
    (0,   0,   0):   0,   # Background
    (0,   255, 0):   1,   # BODY
    (0,   255, 255): 2,   # URINARY BLADDER
    (153, 146, 255): 3,   # SMALL BOWEL
    (64,  64,  128): 4,   # RECTUM
    (255, 255, 0):   5,   # FEMORAL HEAD
    (255, 60,  255): 6,   # GTV
    (255, 55,  55):  7,   # CTV
}

# ===============================
# 3. LOAD MODEL
# ===============================
custom_objects = {
    'combined_loss':        combined_loss,
    'focal_loss':           focal_loss,
    'soft_dice_loss':       soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU':        CustomMeanIoU
}

print("Loading model...")
model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=True)
print("Model loaded.")

# ===============================
# 4. HELPER: decode RGB mask → class ID map
# ===============================
def decode_mask(mask_path):
    """
    Reads a colour PNG mask and converts it to a 2D array of class IDs.
    Pixels with no matching RGB entry are assigned class 0 (Background).
    """
    mask_img = np.array(Image.open(mask_path).convert("RGB"))
    class_map = np.zeros(mask_img.shape[:2], dtype=np.uint8)
    for rgb, cls_id in rgb_to_class.items():
        match = np.all(mask_img == np.array(rgb, dtype=np.uint8), axis=-1)
        class_map[match] = cls_id
    return class_map

# ===============================
# 5. HELPER: predict → class ID map
# ===============================
def predict_mask(image_path):
    """
    Loads an image, runs model inference, returns 2D predicted class ID array
    resized back to the original image dimensions.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    original_h, original_w = img.shape[:2]

    img_bgr     = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, IMAGE_SIZE)
    img_norm    = normalize(np.array([img_resized], dtype=np.float32), axis=1)

    prediction  = model.predict(img_norm, verbose=0)
    pred_class  = np.argmax(prediction, axis=-1)[0]          # (H, W)

    # Resize back to original resolution (nearest neighbour preserves class IDs)
    pred_class  = cv2.resize(pred_class.astype(np.uint8),
                             (original_w, original_h),
                             interpolation=cv2.INTER_NEAREST)
    return pred_class

# ===============================
# 6. HELPER: compute per-class metrics
# ===============================
def compute_metrics(true_flat, pred_flat, filename):
    """
    Computes TP/TN/FP/FN and derived metrics for every class present
    in either the ground truth or the prediction.
    Returns a list of dicts (one per class), matching the original Excel schema.
    """
    total_pixels = len(true_flat)
    rows = []

    present_classes = np.union1d(np.unique(true_flat), np.unique(pred_flat))

    for class_id in present_classes:
        true_bin = (true_flat == class_id)
        pred_bin = (pred_flat == class_id)

        tp = int(np.sum( true_bin &  pred_bin))
        tn = int(np.sum(~true_bin & ~pred_bin))
        fp = int(np.sum(~true_bin &  pred_bin))
        fn = int(np.sum( true_bin & ~pred_bin))

        intersection = tp
        union        = tp + fp + fn

        iou         = tp / union               if union > 0           else 0.0
        dice        = (2*tp) / (2*tp + fp + fn) if (2*tp+fp+fn) > 0  else 0.0
        precision   = tp / (tp + fp)            if (tp + fp) > 0      else 0.0
        recall      = tp / (tp + fn)            if (tp + fn) > 0      else 0.0
        specificity = tn / (tn + fp)            if (tn + fp) > 0      else 0.0
        f1          = (2*precision*recall) / (precision+recall) if (precision+recall) > 0 else 0.0
        tpr         = recall
        fpr         = fp / (fp + tn)            if (fp + tn) > 0      else 0.0

        rows.append({
            'Filename':           filename,
            'Class_ID':           int(class_id),
            'Class_Name':         class_names.get(int(class_id), str(class_id)),
            'Total_Pixels':       total_pixels,
            'TP':                 tp,
            'TN':                 tn,
            'FP':                 fp,
            'FN':                 fn,
            'Intersection':       intersection,
            'Union':              union,
            'IoU':                iou,
            'Dice':               dice,
            'Precision':          precision,
            'Recall':             recall,
            'Specificity':        specificity,
            'F1_Score':           f1,
            'True_Positive_Rate': tpr,
            'False_Positive_Rate':fpr,
        })

    return rows

# ===============================
# 7. MAIN INFERENCE LOOP
# ===============================
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
image_files = sorted([
    f for f in os.listdir(IMAGES_FOLDER)
    if f.lower().endswith(image_extensions)
])

print(f"Found {len(image_files)} images. Starting inference...\n")

all_metrics = []
skipped     = []

for idx, img_file in enumerate(image_files):
    filename   = os.path.splitext(img_file)[0]          # strip extension
    image_path = os.path.join(IMAGES_FOLDER, img_file)
    mask_path  = os.path.join(MASKS_FOLDER,  img_file)  # same filename

    if not os.path.exists(mask_path):
        print(f"  [SKIP] No mask found for: {img_file}")
        skipped.append(img_file)
        continue

    try:
        true_mask = decode_mask(mask_path)
        pred_mask = predict_mask(image_path)

        # Ensure shapes match (safety check)
        if true_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask,
                                   (true_mask.shape[1], true_mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        rows = compute_metrics(true_mask.flatten(), pred_mask.flatten(), filename)
        all_metrics.extend(rows)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(image_files):
            print(f"  Processed {idx+1}/{len(image_files)}: {filename}")

    except Exception as e:
        print(f"  [ERROR] {img_file}: {e}")
        skipped.append(img_file)

print(f"\nInference complete. {len(image_files)-len(skipped)} images processed, {len(skipped)} skipped.")

# ===============================
# 8. BUILD DATAFRAME
# ===============================
column_order = [
    'Filename', 'Class_ID', 'Class_Name', 'Total_Pixels',
    'TP', 'TN', 'FP', 'FN', 'Intersection', 'Union',
    'IoU', 'Dice', 'Precision', 'Recall', 'Specificity', 'F1_Score',
    'True_Positive_Rate', 'False_Positive_Rate'
]

df_detailed = pd.DataFrame(all_metrics)[column_order]

# ===============================
# 9. SAVE EXCEL — identical sheet structure to original
# ===============================
os.makedirs(os.path.dirname(SAVE_EXCEL), exist_ok=True)

with pd.ExcelWriter(SAVE_EXCEL, engine='openpyxl') as writer:

    # ── Sheet 1: Detailed per image per class ──
    df_detailed.to_excel(writer, sheet_name='Detailed_Metrics', index=False)

    # ── Sheet 2: Class summary ──
    class_summary = df_detailed.groupby(['Class_ID', 'Class_Name']).agg(
        TP=('TP','sum'), TN=('TN','sum'), FP=('FP','sum'), FN=('FN','sum'),
        Intersection=('Intersection','sum'), Union=('Union','sum'),
        IoU=('IoU','mean'), Dice=('Dice','mean'),
        Precision=('Precision','mean'), Recall=('Recall','mean'),
        Specificity=('Specificity','mean'), F1_Score=('F1_Score','mean'),
        Total_Pixels=('Total_Pixels','sum')
    ).reset_index()

    for idx, row in class_summary.iterrows():
        tp, fp, fn = row['TP'], row['FP'], row['FN']
        union      = row['Union']
        class_summary.at[idx, 'Aggregated_IoU']  = row['Intersection'] / union if union > 0 else 0.0
        class_summary.at[idx, 'Aggregated_Dice'] = (2*tp) / (2*tp+fp+fn)       if (2*tp+fp+fn) > 0 else 0.0

    class_summary.to_excel(writer, sheet_name='Class_Summary', index=False)

    # ── Sheet 3: Image summary ──
    image_summary = df_detailed.groupby('Filename').agg(
        TP=('TP','sum'), TN=('TN','sum'), FP=('FP','sum'), FN=('FN','sum'),
        IoU=('IoU','mean'), Dice=('Dice','mean'),
        Precision=('Precision','mean'), Recall=('Recall','mean'),
        F1_Score=('F1_Score','mean'),
        Total_Pixels=('Total_Pixels','first'),
        Classes_Present=('Class_ID','count')
    ).reset_index()

    denom_iou  = (image_summary['TP'] + image_summary['FP'] + image_summary['FN']).replace(0, 1)
    denom_dice = (2*image_summary['TP'] + image_summary['FP'] + image_summary['FN']).replace(0, 1)
    image_summary['Image_IoU']  = image_summary['TP'] / denom_iou
    image_summary['Image_Dice'] = (2 * image_summary['TP']) / denom_dice

    image_summary.to_excel(writer, sheet_name='Image_Summary', index=False)

    # ── Sheet 4: Class presence ──
    class_presence = df_detailed.groupby(['Class_ID', 'Class_Name']).agg(
        Images_Present=('Filename','count'),
        TP=('TP','sum'),
        FN=('FN','sum')
    ).reset_index()
    class_presence['Support'] = class_presence['TP'] + class_presence['FN']
    class_presence.to_excel(writer, sheet_name='Class_Presence', index=False)

print(f"\nExcel saved to: {SAVE_EXCEL}")
print(f"Total images processed : {df_detailed['Filename'].nunique()}")
print(f"Total records          : {len(df_detailed)}")
print(f"\nOverall Performance:")
print(f"  Mean IoU  : {df_detailed['IoU'].mean():.4f}")
print(f"  Mean Dice : {df_detailed['Dice'].mean():.4f}")
print(f"\nClass Presence Summary:")
total_imgs = df_detailed['Filename'].nunique()
for cid, cname in class_names.items():
    n = len(df_detailed[df_detailed['Class_ID'] == cid])
    if n > 0:
        print(f"  {cname:20s}: {n}/{total_imgs} ({100*n/total_imgs:.1f}%)")
if skipped:
    print(f"\nSkipped files ({len(skipped)}): {skipped}")




















































#Qualitative Analysis


# ========== Prediction =========================    
    
# ========== Prediction =========================    
    
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import normalize
import tensorflow as tf
from keras.saving import register_keras_serializable
from PIL import Image
import matplotlib.pyplot as plt
import random

# Define custom loss and metric functions (from your original code)
@register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)
    loss = tf.pow(1 - y_pred, gamma) * cross_entropy
    return tf.reduce_mean(loss, axis=-1)

@register_keras_serializable()
def soft_dice_loss(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    sum_true = tf.reduce_sum(y_true, axis=(1, 2, 3))
    sum_pred = tf.reduce_sum(y_pred, axis=(1, 2, 3))
    dice_coefficient = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
    dice_loss = 1 - dice_coefficient
    return tf.reduce_mean(dice_loss)

@register_keras_serializable()
def combined_loss(y_true, y_pred, gamma=2.0, alpha=0.5):
    focal = focal_loss(y_true, y_pred, gamma)
    dice = soft_dice_loss(y_true, y_pred)
    return alpha * focal + (1 - alpha) * dice

@register_keras_serializable()
class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    def _init_(self, num_classes, name='mean_iou', dtype='float32', ignore_class=None,
                 sparse_y_true=True, sparse_y_pred=True, axis=-1):
        super(CustomMeanIoU, self)._init_(num_classes=num_classes, name=name, dtype=dtype, ignore_class=ignore_class)
        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update({'sparse_y_true': self.sparse_y_true, 'sparse_y_pred': self.sparse_y_pred, 'axis': self.axis})
        return config

# Load the model once
model_path = r"I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/UNET_BM/134 epochs/best_model.keras"
custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}
model = load_model(model_path, custom_objects=custom_objects, compile=True)

# Class RGB values

class_rgb_values = {
    0: (0, 0, 0),          # Background (black)
    1: (0, 255, 0),        # BODY
    2: (0, 255, 255),      # URINARY BLADDER
    3: (153, 146, 255),      # SMALL BOWEL
    4: (64, 64, 128),      # RECTUM
    5: (255, 255, 0),      # FEMORAL HEAD
    6: (255, 60, 255),     # GTV
    7: (255, 55, 55),      # CTV
}

# (0, 255, 0)	#00FF00          = body
# (0, 255, 255)	#00FFFF          = Uninary
# (64, 64, 128)	#404080          = Rectum
# (153, 146, 255)#9992FF         = Small Bowel
# (255, 55, 55)	#FF3737          = CTV
# (255, 60, 255)#FF3CFF          = GTV
# (255, 255, 0)	#FFFF00          = Femoral Head
def segment_image(image, image_size=(512, 512)):
    # Convert PIL image to numpy array
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    if img is None:
        raise ValueError("Could not load image")
   
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, image_size)
    img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)
    prediction = model.predict(img_normalized)
    predicted_mask = np.argmax(prediction, axis=-1)[0]
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create RGB mask
    rgb_mask = np.zeros((*predicted_mask_resized.shape, 3), dtype=np.uint8)
    for class_idx, rgb in class_rgb_values.items():
        rgb_mask[predicted_mask_resized == class_idx] = rgb

    # Convert back to PIL for visualization
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_mask)

def save_plot_in_vector_format(fig, save_path, format='eps'):
    """
    Save the plot in a vector format (e.g., EPS or SVG).
    
    Args:
        fig: The Matplotlib figure object to save.
        save_path: Path (including filename) where the plot will be saved.
        format: Format to save the plot ('eps' or 'svg'). Default is 'eps'.
    """
    if format not in ['eps', 'svg']:
        raise ValueError("Unsupported format. Use 'eps' or 'svg'.")
    
    # Save the figure in the specified vector format
    fig.savefig(save_path, format=format, bbox_inches='tight', dpi=600)
    print(f"Plot saved in {format.upper()} format at: {save_path}")


def visualize_random_image_and_mask(images_folder, masks_folder, image_size=(512, 512), save_path=None):
    """
    Visualize a random image, true mask, and predicted mask.
    Optionally save the plot in a vector format.
    
    Args:
        images_folder: Path to the folder containing input images.
        masks_folder: Path to the folder containing true masks.
        image_size: Size to which images are resized (default: (256, 256)).
        save_path: Path to save the plot in vector format (optional).
    """
    # Get list of images and masks
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No images found in the images folder.")
        return

    # Select a random image
    random_image_file = random.choice(image_files)
    print(f"Selected random image: {random_image_file}")

    # Load the image and corresponding mask
    image_path = os.path.join(images_folder, random_image_file)
    mask_path = os.path.join(masks_folder, random_image_file)  # Assuming mask has the same filename

    image = Image.open(image_path)
    true_mask = Image.open(mask_path)

    # Predict the segmentation mask
    predicted_mask = segment_image(image, image_size)

    # Create the figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Original Image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image", fontsize=7)
    axes[0].axis('off')

    # True Mask
    axes[1].imshow(true_mask, cmap='cividis')
    axes[1].set_title("True Mask", fontsize=7)
    axes[1].axis('off')

    # Predicted Mask
    axes[2].imshow(predicted_mask, cmap='cividis')
    axes[2].set_title("Predicted Mask", fontsize=7)
    axes[2].axis('off')

    plt.tight_layout()

    # Save the plot in vector format if save_path is provided
    if save_path:
        save_plot_in_vector_format(fig, save_path, format='eps')  # Change format to 'svg' if needed

    plt.show()


# Example usage
images_folder = r"I:\Radiotherapy\Cervix\cervix_small_set\images"  # Replace with your images folder path
masks_folder = r"I:\Radiotherapy\Cervix\cervix_small_set\masks"  # Replace with your masks folder path

# Define the path to save the plot in vector format
save_path = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/UNET_Small_512\random_image_and_mask_visualization.eps"  # Change to 'random_image_and_mask_visualization.svg' for SVG format

# Visualize and save the plot
visualize_random_image_and_mask(images_folder, masks_folder, save_path=save_path)



















# Error Analysis Map Entire Folder pdf eps png


#----------- Prediction with Error Map Entire Folder  Final with Editable PDF Text ----------------------------

import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import normalize
import tensorflow as tf
from keras.saving import register_keras_serializable
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

image_size = (512,512)

# Configure matplotlib for editable text in PDF
matplotlib.rcParams['pdf.fonttype'] = 42  # 42 = TrueType fonts (editable)
matplotlib.rcParams['ps.fonttype'] = 42   # For PostScript (EPS) as well
matplotlib.rcParams['font.family'] = 'sans-serif'  # Use standard fonts
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']  # Common editable fonts

# [Previous custom loss and metric functions remain the same]
# ... (keeping your existing custom functions unchanged)

# Load the model
model_path = r"I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/UNET_BM/200 Epochs/best_model.keras"
custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}
model = load_model(model_path, custom_objects=custom_objects, compile=True)

# Class RGB values
class_rgb_values = {
    0: (0, 0, 0),          # Background (black)
    1: (0, 255, 0),        # BODY
    2: (0, 255, 255),      # URINARY BLADDER
    3: (153, 146, 255),      # SMALL BOWEL
    4: (64, 64, 128),      # RECTUM
    5: (255, 255, 0),      # FEMORAL HEAD
    6: (255, 60, 255),     # GTV
    7: (255, 55, 55),      # CTV
}

def segment_image(image, image_size):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError("Could not load image")
   
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, image_size)
    img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)
    prediction = model.predict(img_normalized)
    predicted_mask = np.argmax(prediction, axis=-1)[0]
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
    
    # Create RGB mask for visualization
    rgb_mask = np.zeros((*predicted_mask_resized.shape, 3), dtype=np.uint8)
    for class_idx, rgb in class_rgb_values.items():
        rgb_mask[predicted_mask_resized == class_idx] = rgb
    
    return Image.fromarray(cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)), predicted_mask_resized


def create_error_map_overlay(true_mask, predicted_mask, original_image, alpha=0.5):
    """
    Create error map overlay for multi-class segmentation.
    - Green: Correctly predicted (all non-background classes)
    - Red: False Negatives (missed or wrong class)
    - Blue: False Positives (predicted class where should be background)
    - No overlay: True Negatives (correctly predicted background)
    """
    true_mask_array = np.array(true_mask)
    
    # Convert RGB mask to class indices
    if len(true_mask_array.shape) == 3:
        # CRITICAL FIX: Convert true mask from RGB to BGR to match class_rgb_values
        true_mask_bgr = cv2.cvtColor(true_mask_array, cv2.COLOR_RGB2BGR)
        
        true_mask_indices = np.zeros(true_mask_array.shape[:2], dtype=np.uint8)
        for class_idx, bgr in class_rgb_values.items():
            # Now comparing BGR with BGR
            mask = np.all(true_mask_bgr == bgr, axis=-1)
            true_mask_indices[mask] = class_idx
            
            # Debug: print how many pixels found for each class
            if np.sum(mask) > 0:
                print(f"  Class {class_idx}: {np.sum(mask)} pixels")
    else:
        true_mask_indices = true_mask_array
    
    # Correct predictions
    correct_mask = (true_mask_indices == predicted_mask)
    tp_mask = correct_mask & (true_mask_indices > 0)  # Correct non-background
    tn_mask = correct_mask & (true_mask_indices == 0)  # Correct background
    
    # Wrong predictions
    wrong_mask = (true_mask_indices != predicted_mask)
    
    # False Negatives: true mask has a class, but prediction is wrong
    fn_mask = wrong_mask & (true_mask_indices > 0)
    
    # False Positives: prediction has a class, but true mask is background
    fp_mask = wrong_mask & (predicted_mask > 0) & (true_mask_indices == 0)
    
    # Debug: Print counts and percentages
    total_pixels = true_mask_indices.size
    print(f"True Positives (Correct): {np.sum(tp_mask)} pixels ({np.sum(tp_mask)/total_pixels*100:.2f}%)")
    print(f"False Negatives (Missed/Wrong class): {np.sum(fn_mask)} pixels ({np.sum(fn_mask)/total_pixels*100:.2f}%)")
    print(f"False Positives (Background predicted as class): {np.sum(fp_mask)} pixels ({np.sum(fp_mask)/total_pixels*100:.2f}%)")
    print(f"True Negatives (Correct background): {np.sum(tn_mask)} pixels ({np.sum(tn_mask)/total_pixels*100:.2f}%)")
    print(f"Overall Accuracy: {(np.sum(tp_mask) + np.sum(tn_mask))/total_pixels*100:.2f}%")
    
    # Create error map - RGB format for display
    error_map = np.zeros((*true_mask_indices.shape, 3), dtype=np.uint8)
    error_map[tp_mask] = [0, 255, 0]    # TP: Green in RGB
    error_map[fn_mask] = [255, 0, 0]    # FN: Red in RGB
    error_map[fp_mask] = [0, 0, 255]    # FP: Blue in RGB
    # TN remains black (no overlay for background)
    
    # Create overlay - start with original image
    original_array = np.array(original_image)
    if len(original_array.shape) == 2:
        original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
    
    overlay = original_array.copy()
    
    # Overlay all non-background areas (TP, FN, FP) - exclude TN
    overlay_mask = tp_mask | fn_mask | fp_mask
    if np.sum(overlay_mask) > 0:  # Only blend if there are areas to overlay
        overlay[overlay_mask] = (alpha * error_map[overlay_mask] + 
                                  (1 - alpha) * original_array[overlay_mask]).astype(np.uint8)
    
    return Image.fromarray(overlay)



def save_plot_in_vector_format(fig, save_path, formats=['eps', 'png', 'pdf']):
    """
    Save the plot in multiple formats (e.g., EPS, PNG, and PDF).
    Text will be editable in PDF when opened in Illustrator.
    
    Args:
        fig: The Matplotlib figure object to save.
        save_path: Path (including filename without extension) where the plot will be saved.
        formats: List of formats to save the plot (e.g., ['eps', 'png', 'pdf']). 
                 Default is ['eps', 'png', 'pdf'].
    """
    for fmt in formats:
        if fmt not in ['eps', 'svg', 'png', 'pdf']:
            raise ValueError(f"Unsupported format: {fmt}. Use 'eps', 'svg', 'png', or 'pdf'.")
        
        full_path = f"{save_path}.{fmt}"
        
        if fmt == 'pdf':
            # For PDF with editable text
            fig.savefig(full_path, format='pdf', bbox_inches='tight', 
                       dpi=600, metadata={'Creator': 'Matplotlib'})
        elif fmt == 'eps':
            # For EPS with editable text
            fig.savefig(full_path, format='eps', bbox_inches='tight', 
                       dpi=600)
        else:  # For PNG
            fig.savefig(full_path, format=fmt, bbox_inches='tight', dpi=600)
        
        print(f"Plot saved in {fmt.upper()} format at: {full_path}")

def visualize_all_four_plots(images_folder, masks_folder, save_dir, image_size, alpha=0.5):
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No images found in the images folder.")
        return

    for image_file in image_files:
        print(f"Processing image: {image_file}")

        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, image_file)

        try:
            image = Image.open(image_path)
            true_mask = Image.open(mask_path)
            predicted_mask_img, predicted_mask_array = segment_image(image, image_size)
            overlay_image = create_error_map_overlay(true_mask, predicted_mask_array, image, alpha)

            # Visualize four plots side by side
            fig = plt.figure(figsize=(20, 5))

            # Original Image
            plt.subplot(1, 4, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            # True Mask
            plt.subplot(1, 4, 2)
            plt.title("True Mask")
            plt.imshow(true_mask, cmap='cividis')
            plt.axis('off')

            # Predicted Mask
            plt.subplot(1, 4, 3)
            plt.title("Predicted Mask")
            plt.imshow(predicted_mask_img, cmap='cividis')
            plt.axis('off')

            # Error Map Overlay
            plt.subplot(1, 4, 4)
            plt.title(f"Error Map Overlay (alpha={alpha})")
            plt.imshow(overlay_image)
            plt.axis('off')
            plt.show()

            plt.tight_layout()

            # Save the plot with the same name as the image in EPS, PNG, and PDF formats
            image_name = os.path.splitext(image_file)[0]
            save_path = os.path.join(save_dir, image_name)
            save_plot_in_vector_format(fig, save_path, formats=['png', 'pdf'])

            plt.close(fig)  # Close the figure to free memory

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

# Example usage
images_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/images"  # Replace with your images folder path
masks_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/masks"  # Replace with your masks folder path
save_dir = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET_Small_512_200_epoch"

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Adjust alpha value here (0.0 to 1.0)
visualize_all_four_plots(images_folder, masks_folder, save_dir,image_size, alpha=0.5)











































#----------- Prediction with Error Map Entire Folder  Final with Editable PDF Text BAT-RM ----------------------------


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

# ===============================
# 1. MODEL ARCHITECTURE — BAT-RM
# ===============================
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)


class GatedBATBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.sobel_x = nn.Parameter(
            torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                         dtype=torch.float32).view(1,1,3,3), requires_grad=False)
        self.sobel_y = nn.Parameter(
            torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                         dtype=torch.float32).view(1,1,3,3), requires_grad=False)
        self.query = nn.Conv2d(in_ch, in_ch//8, 1)
        self.key   = nn.Conv2d(in_ch, in_ch//8, 1)
        self.value = nn.Conv2d(in_ch, in_ch,    1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        gray   = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(gray, self.sobel_x, padding=1)
        grad_y = F.conv2d(gray, self.sobel_y, padding=1)
        gate   = torch.sigmoid(torch.sqrt(grad_x**2 + grad_y**2 + 1e-6))
        b, c, h, w = x.size()
        q    = self.query(x*gate).view(b,-1,h*w).permute(0,2,1)
        k    = self.key(x*gate).view(b,-1,h*w)
        v    = self.value(x).view(b,-1,h*w)
        attn = F.softmax(torch.bmm(q,k), dim=-1)
        out  = torch.bmm(v, attn.permute(0,2,1)).view(b,c,h,w)
        return self.gamma * out + x, gate


class RegionMambaBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch)
        self.ssm  = nn.Linear(in_ch, in_ch)
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)

    def forward(self, x):
        shortcut = x
        b, c, h, w = x.shape
        x_flat = x.permute(0,2,3,1).reshape(b,-1,c)
        x_flat = self.norm(x_flat)
        x = x_flat.reshape(b,h,w,c).permute(0,3,1,2)
        x = self.conv(x)
        x = x.permute(0,2,3,1).reshape(b,-1,c)
        x = self.ssm(x)
        return x.reshape(b,h,w,c).permute(0,3,1,2) + shortcut


class BRAFModule(nn.Module):
    def __init__(self, bat_ch=128, rm_ch=512):
        super().__init__()
        self.rm_project = nn.Conv2d(rm_ch, bat_ch, 1)
        self.alpha_conv = nn.Sequential(
            nn.Conv2d(bat_ch+bat_ch, 1, 3, padding=1), nn.Sigmoid())
        self.refine = nn.Conv2d(bat_ch, bat_ch, 3, padding=1)

    def forward(self, f_bat, f_rm, gate):
        f_rm_up      = F.interpolate(f_rm, size=f_bat.shape[2:],
                                     mode='bilinear', align_corners=False)
        f_rm_aligned = self.rm_project(f_rm_up)
        alpha        = self.alpha_conv(torch.cat([f_bat, f_rm_aligned], dim=1))
        f_fuse       = alpha*f_bat + (1-alpha)*f_rm_aligned
        return self.refine(f_fuse) * gate


class BAT_RM_UNet(nn.Module):
    def __init__(self, n_classes, in_channels=3):
        super().__init__()
        self.e1=EncoderBlock(in_channels,32); self.e2=EncoderBlock(32,64)
        self.e3=EncoderBlock(64,128);         self.e4=EncoderBlock(128,256)
        self.e5=EncoderBlock(256,512);        self.pool=nn.MaxPool2d(2)
        self.bat=GatedBATBlock(128); self.rm=RegionMambaBlock(512)
        self.braf=BRAFModule(128,512)
        self.up5=nn.ConvTranspose2d(512,256,2,2); self.d5=EncoderBlock(512,256)
        self.up4=nn.ConvTranspose2d(256,128,2,2); self.d4=EncoderBlock(256,128)
        self.up3=nn.ConvTranspose2d(128,64,2,2);  self.d3=EncoderBlock(192,64)
        self.up2=nn.ConvTranspose2d(64,32,2,2);   self.d2=EncoderBlock(96,32)
        self.out_conv=nn.Conv2d(32,n_classes,1)

    def forward(self, x):
        s1=self.e1(x); s2=self.e2(self.pool(s1))
        s3=self.e3(self.pool(s2)); s4=self.e4(self.pool(s3))
        s5=self.e5(self.pool(s4))
        f_bat,gate=self.bat(s3); f_rm=self.rm(s5)
        f_fuse=self.braf(f_bat,f_rm,gate)
        x5=self.up5(s5)
        if x5.shape[2:]!=s4.shape[2:]: x5=F.interpolate(x5,size=s4.shape[2:],mode='bilinear')
        x5=self.d5(torch.cat([x5,s4],dim=1))
        x4=self.up4(x5)
        if x4.shape[2:]!=s3.shape[2:]: x4=F.interpolate(x4,size=s3.shape[2:],mode='bilinear')
        x4=self.d4(torch.cat([x4,s3],dim=1))
        x3=self.up3(x4)
        f_fuse_up=F.interpolate(f_fuse,size=x3.shape[2:],mode='bilinear')
        x3=self.d3(torch.cat([x3,f_fuse_up],dim=1))
        x2=self.up2(x3)
        if x2.shape[2:]!=s2.shape[2:]: x2=F.interpolate(x2,size=s2.shape[2:],mode='bilinear')
        x2=self.d2(torch.cat([x2,s2],dim=1))
        return self.out_conv(F.interpolate(x2,size=x.shape[2:],mode='bilinear'))

# ===============================
# 2. CONFIGURATION
# ===============================
MODEL_PATH    = r'I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/BAT-RM/Cervix_small_Axial_100_epochs_pytorch_best.pth'
IMAGES_FOLDER = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/images'
MASKS_FOLDER  = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/masks'
SAVE_DIR      = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/BAT-RM'

IMAGE_SIZE = (512, 512)   # (W, H) — must match training
ALPHA      = 0.5          # overlay transparency

class_names = {
    0: 'Background', 1: 'BODY',      2: 'URINARY BLADDER',
    3: 'SMALL BOWEL', 4: 'RECTUM',   5: 'FEMORAL HEAD',
    6: 'GTV',         7: 'CTV',
}
n_classes = len(class_names)

# Verified BGR mapping (matches training label_encode_mask behaviour)
bgr_to_class = {
    (0,   0,   0):   0,
    (0,   255, 0):   1,
    (0,   255, 255): 2,
    (153, 146, 255): 3,
    (64,  64,  128): 4,
    (255, 255, 0):   5,
    (255, 60,  255): 6,
    (255, 55,  55):  7,
}

# Class ID → BGR colour for predicted mask visualisation
class_bgr_values = {v: k for k, v in bgr_to_class.items()}

# ===============================
# 3. LOAD MODEL
# ===============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

model      = BAT_RM_UNet(n_classes=n_classes, in_channels=3).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint (epoch: {checkpoint.get('epoch','unknown')})")
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()
print(f"Model loaded: {MODEL_PATH}")

# ===============================
# 4. HELPERS
# ===============================
def decode_mask(mask_path):
    """BGR mask PNG → 2D class-ID array."""
    mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_bgr is None:
        raise ValueError(f"Cannot load mask: {mask_path}")
    class_map = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    for bgr, cls_id in bgr_to_class.items():
        class_map[np.all(mask_bgr == np.array(bgr, dtype=np.uint8), axis=-1)] = cls_id
    return class_map


def predict_mask(image_path):
    """BGR image → 2D predicted class-ID array at original resolution."""
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot load image: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]
    img_resized    = cv2.resize(img_bgr, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    img_norm       = img_resized.astype(np.float32) / 255.0
    img_tensor     = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits     = model(img_tensor)
        pred_class = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(pred_class, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def class_map_to_bgr(class_map):
    """2D class-ID array → BGR colour image for display."""
    bgr = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, colour in class_bgr_values.items():
        bgr[class_map == cls_id] = colour
    return bgr


def create_error_map_overlay(true_class_map, pred_class_map, original_bgr, alpha=0.5):
    """
    Builds an error-map overlay on the original image.
      Green  = True Positive  (correct non-background)
      Red    = False Negative (missed / wrong class)
      Blue   = False Positive (predicted class where GT is background)
      Black  = True Negative  (correct background — no overlay)
    Returns RGB PIL Image.
    """
    correct  = (true_class_map == pred_class_map)
    tp_mask  = correct  & (true_class_map > 0)
    tn_mask  = correct  & (true_class_map == 0)
    wrong    = ~correct
    fn_mask  = wrong    & (true_class_map > 0)
    fp_mask  = wrong    & (pred_class_map > 0) & (true_class_map == 0)

    total = true_class_map.size
    print(f"  TP (green) : {tp_mask.sum():>8,}  ({100*tp_mask.sum()/total:.2f}%)")
    print(f"  FN (red)   : {fn_mask.sum():>8,}  ({100*fn_mask.sum()/total:.2f}%)")
    print(f"  FP (blue)  : {fp_mask.sum():>8,}  ({100*fp_mask.sum()/total:.2f}%)")
    print(f"  TN (none)  : {tn_mask.sum():>8,}  ({100*tn_mask.sum()/total:.2f}%)")
    print(f"  Accuracy   : {100*(tp_mask.sum()+tn_mask.sum())/total:.2f}%")

    # Error colour map (RGB)
    error_rgb = np.zeros((*true_class_map.shape, 3), dtype=np.uint8)
    error_rgb[tp_mask] = [0,   255, 0  ]   # green
    error_rgb[fn_mask] = [255, 0,   0  ]   # red
    error_rgb[fp_mask] = [0,   0,   255]   # blue

    # Convert original BGR → RGB for blending
    orig_rgb    = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    overlay     = orig_rgb.copy()
    blend_mask  = tp_mask | fn_mask | fp_mask
    if blend_mask.any():
        overlay[blend_mask] = (
            alpha * error_rgb[blend_mask] +
            (1 - alpha) * orig_rgb[blend_mask]
        ).astype(np.uint8)

    return Image.fromarray(overlay)


def save_figure(fig, save_path_no_ext, formats=('png', 'pdf')):
    for fmt in formats:
        path = f"{save_path_no_ext}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=600)
        print(f"  Saved {fmt.upper()}: {path}")


# ===============================
# 5. MAIN LOOP
# ===============================
os.makedirs(SAVE_DIR, exist_ok=True)

image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
image_files = sorted([
    f for f in os.listdir(IMAGES_FOLDER)
    if f.lower().endswith(image_extensions)
])
print(f"\nFound {len(image_files)} images. Starting...\n")

for img_file in image_files:
    print(f"Processing: {img_file}")

    image_path = os.path.join(IMAGES_FOLDER, img_file)
    mask_path  = os.path.join(MASKS_FOLDER,  img_file)

    if not os.path.exists(mask_path):
        print(f"  [SKIP] No mask found.")
        continue

    try:
        # ── Load ──
        orig_bgr       = cv2.imread(image_path, cv2.IMREAD_COLOR)
        orig_rgb       = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        true_class_map = decode_mask(mask_path)
        pred_class_map = predict_mask(image_path)

        # Safety: shape match
        if true_class_map.shape != pred_class_map.shape:
            pred_class_map = cv2.resize(pred_class_map,
                                        (true_class_map.shape[1], true_class_map.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

        # ── Colour masks for display ──
        true_bgr  = class_map_to_bgr(true_class_map)
        true_rgb  = cv2.cvtColor(true_bgr, cv2.COLOR_BGR2RGB)
        pred_bgr  = class_map_to_bgr(pred_class_map)
        pred_rgb  = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)

        # ── Error map overlay ──
        overlay_pil = create_error_map_overlay(
            true_class_map, pred_class_map, orig_bgr, alpha=ALPHA)

        # ── Plot ──
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(os.path.splitext(img_file)[0], fontsize=10)

        axes[0].imshow(orig_rgb);    axes[0].set_title("Original Image"); axes[0].axis('off')
        axes[1].imshow(true_rgb);    axes[1].set_title("True Mask");      axes[1].axis('off')
        axes[2].imshow(pred_rgb);    axes[2].set_title("Predicted Mask"); axes[2].axis('off')
        axes[3].imshow(overlay_pil); axes[3].set_title(f"Error Map (α={ALPHA})"); axes[3].axis('off')

        plt.tight_layout()

        # ── Save ──
        save_name = os.path.join(SAVE_DIR, os.path.splitext(img_file)[0])
        save_figure(fig, save_name, formats=['png', 'pdf'])
        plt.close(fig)

    except Exception as e:
        print(f"  [ERROR] {img_file}: {e}")
        continue

print("\nDone.")















































































#----------- Prediction with Error Map Entire Folder  Final with Editable PDF Text nnUNet ----------------------------

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

# ===============================
# 1. MODEL ARCHITECTURE — nnUNet
# ===============================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )
        self.conv = ConvBlock(out_channels, out_channels)
    def forward(self, x):
        return self.conv(self.down(x))


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class NNUNet2D(nn.Module):
    def __init__(self, n_classes, in_channels=3, base_features=32):
        super().__init__()
        f = base_features
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = Down(f,      f * 2)
        self.enc3 = Down(f * 2,  f * 4)
        self.enc4 = Down(f * 4,  f * 8)
        self.enc5 = Down(f * 8,  f * 16)
        self.up4  = Up(f * 16,   f * 8,  f * 8)
        self.up3  = Up(f * 8,    f * 4,  f * 4)
        self.up2  = Up(f * 4,    f * 2,  f * 2)
        self.up1  = Up(f * 2,    f,      f)
        self.out1 = nn.Conv2d(f,      n_classes, 1)
        self.out2 = nn.Conv2d(f * 2,  n_classes, 1)
        self.out3 = nn.Conv2d(f * 4,  n_classes, 1)
        self.out4 = nn.Conv2d(f * 8,  n_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        b  = self.enc5(s4)
        d4 = self.up4(b,  s4)
        d3 = self.up3(d4, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        out_main = self.out1(d1)
        self.ds_outputs = [
            out_main,
            F.interpolate(self.out2(d2), size=out_main.shape[-2:], mode='bilinear', align_corners=False),
            F.interpolate(self.out3(d3), size=out_main.shape[-2:], mode='bilinear', align_corners=False),
            F.interpolate(self.out4(d4), size=out_main.shape[-2:], mode='bilinear', align_corners=False),
        ]
        return out_main

# ===============================
# 2. CONFIGURATION
# ===============================
MODEL_PATH    = r'I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/nnUNet_256/Cervix_small_Axial_100_epochs_pytorch_best_nnUNET_BM.pth'
IMAGES_FOLDER = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/images'
MASKS_FOLDER  = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/masks'
SAVE_DIR      = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/nnUNet'

IMAGE_SIZE = (256, 256)   # (W, H) — must match training
ALPHA      = 0.5

class_names = {
    0: 'Background', 1: 'BODY',      2: 'URINARY BLADDER',
    3: 'SMALL BOWEL', 4: 'RECTUM',   5: 'FEMORAL HEAD',
    6: 'GTV',         7: 'CTV',
}
n_classes = len(class_names)

bgr_to_class = {
    (0,   0,   0):   0,
    (0,   255, 0):   1,
    (0,   255, 255): 2,
    (153, 146, 255): 3,
    (64,  64,  128): 4,
    (255, 255, 0):   5,
    (255, 60,  255): 6,
    (255, 55,  55):  7,
}
class_bgr_values = {v: k for k, v in bgr_to_class.items()}

# ===============================
# 3. LOAD MODEL
# ===============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

model      = NNUNet2D(n_classes=n_classes, in_channels=3).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded checkpoint (state_dict key)")
else:
    model.load_state_dict(checkpoint)
    print("Loaded raw state dict")

model.eval()
print(f"Model loaded: {MODEL_PATH}")

# ===============================
# 4. HELPERS
# ===============================
def decode_mask(mask_path):
    """BGR mask PNG → 2D class-ID array."""
    mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask_bgr is None:
        raise ValueError(f"Cannot load mask: {mask_path}")
    class_map = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    for bgr, cls_id in bgr_to_class.items():
        class_map[np.all(mask_bgr == np.array(bgr, dtype=np.uint8), axis=-1)] = cls_id
    return class_map


def predict_mask(image_path):
    """BGR image → 2D predicted class-ID array at original resolution."""
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot load image: {image_path}")
    orig_h, orig_w = img_bgr.shape[:2]

    # BGR → RGB to match training dataloader convention
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    img_norm    = img_resized.astype(np.float32) / 255.0
    img_tensor  = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits     = model(img_tensor)
        pred_class = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return cv2.resize(pred_class, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def class_map_to_bgr(class_map):
    """2D class-ID array → BGR colour image for display."""
    bgr = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, colour in class_bgr_values.items():
        bgr[class_map == cls_id] = colour
    return bgr


def create_error_map_overlay(true_class_map, pred_class_map, original_bgr, alpha=0.5):
    """
    Builds an error-map overlay on the original image.
      Green = True Positive  (correct non-background)
      Red   = False Negative (missed / wrong class)
      Blue  = False Positive (predicted class where GT is background)
      Black = True Negative  (correct background — no overlay)
    Returns RGB PIL Image.
    """
    correct = (true_class_map == pred_class_map)
    tp_mask = correct  & (true_class_map > 0)
    tn_mask = correct  & (true_class_map == 0)
    fn_mask = ~correct & (true_class_map > 0)
    fp_mask = ~correct & (pred_class_map > 0) & (true_class_map == 0)

    total = true_class_map.size
    print(f"  TP (green) : {tp_mask.sum():>8,}  ({100*tp_mask.sum()/total:.2f}%)")
    print(f"  FN (red)   : {fn_mask.sum():>8,}  ({100*fn_mask.sum()/total:.2f}%)")
    print(f"  FP (blue)  : {fp_mask.sum():>8,}  ({100*fp_mask.sum()/total:.2f}%)")
    print(f"  TN (none)  : {tn_mask.sum():>8,}  ({100*tn_mask.sum()/total:.2f}%)")
    print(f"  Accuracy   : {100*(tp_mask.sum()+tn_mask.sum())/total:.2f}%")

    error_rgb          = np.zeros((*true_class_map.shape, 3), dtype=np.uint8)
    error_rgb[tp_mask] = [0,   255, 0  ]   # green
    error_rgb[fn_mask] = [255, 0,   0  ]   # red
    error_rgb[fp_mask] = [0,   0,   255]   # blue

    orig_rgb   = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    overlay    = orig_rgb.copy()
    blend_mask = tp_mask | fn_mask | fp_mask
    if blend_mask.any():
        overlay[blend_mask] = (
            alpha * error_rgb[blend_mask] +
            (1 - alpha) * orig_rgb[blend_mask]
        ).astype(np.uint8)

    return Image.fromarray(overlay)


def save_figure(fig, save_path_no_ext, formats=('png', 'pdf')):
    for fmt in formats:
        path = f"{save_path_no_ext}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=600)
        print(f"  Saved {fmt.upper()}: {path}")


# ===============================
# 5. MAIN LOOP
# ===============================
os.makedirs(SAVE_DIR, exist_ok=True)

image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
image_files = sorted([
    f for f in os.listdir(IMAGES_FOLDER)
    if f.lower().endswith(image_extensions)
])
print(f"\nFound {len(image_files)} images. Starting...\n")

for img_file in image_files:
    print(f"Processing: {img_file}")

    image_path = os.path.join(IMAGES_FOLDER, img_file)
    mask_path  = os.path.join(MASKS_FOLDER,  img_file)

    if not os.path.exists(mask_path):
        print(f"  [SKIP] No mask found.")
        continue

    try:
        orig_bgr       = cv2.imread(image_path, cv2.IMREAD_COLOR)
        orig_rgb       = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        true_class_map = decode_mask(mask_path)
        pred_class_map = predict_mask(image_path)

        if true_class_map.shape != pred_class_map.shape:
            pred_class_map = cv2.resize(
                pred_class_map,
                (true_class_map.shape[1], true_class_map.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        true_rgb    = cv2.cvtColor(class_map_to_bgr(true_class_map), cv2.COLOR_BGR2RGB)
        pred_rgb    = cv2.cvtColor(class_map_to_bgr(pred_class_map), cv2.COLOR_BGR2RGB)
        overlay_pil = create_error_map_overlay(true_class_map, pred_class_map, orig_bgr, alpha=ALPHA)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(os.path.splitext(img_file)[0], fontsize=10)
        axes[0].imshow(orig_rgb);    axes[0].set_title("Original Image"); axes[0].axis('off')
        axes[1].imshow(true_rgb);    axes[1].set_title("True Mask");      axes[1].axis('off')
        axes[2].imshow(pred_rgb);    axes[2].set_title("Predicted Mask"); axes[2].axis('off')
        axes[3].imshow(overlay_pil); axes[3].set_title(f"Error Map (α={ALPHA})"); axes[3].axis('off')
        plt.tight_layout()

        save_name = os.path.join(SAVE_DIR, os.path.splitext(img_file)[0])
        save_figure(fig, save_name, formats=['png', 'pdf'])
        plt.close(fig)

    except Exception as e:
        print(f"  [ERROR] {img_file}: {e}")
        continue

print("\nDone.")














































































# Error Analysis Map Entire Folder pdf eps png


#----------- Prediction with Error Map Entire Folder  Final with Editable PDF Text ----------------------------

import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import normalize
import tensorflow as tf
from keras.saving import register_keras_serializable
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

image_size = (512,512)

# Configure matplotlib for editable text in PDF
matplotlib.rcParams['pdf.fonttype'] = 42  # 42 = TrueType fonts (editable)
matplotlib.rcParams['ps.fonttype'] = 42   # For PostScript (EPS) as well
matplotlib.rcParams['font.family'] = 'sans-serif'  # Use standard fonts
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']  # Common editable fonts

# [Previous custom loss and metric functions remain the same]
# ... (keeping your existing custom functions unchanged)

# Load the model
model_path = r"I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/UNET_BM/188 Epochs_best/best_model.keras"
custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}
model = load_model(model_path, custom_objects=custom_objects, compile=True)

# Class RGB values
class_rgb_values = {
    0: (0, 0, 0),          # Background (black)
    1: (0, 255, 0),        # BODY
    2: (0, 255, 255),      # URINARY BLADDER
    3: (153, 146, 255),      # SMALL BOWEL
    4: (64, 64, 128),      # RECTUM
    5: (255, 255, 0),      # FEMORAL HEAD
    6: (255, 60, 255),     # GTV
    7: (255, 55, 55),      # CTV
}

def segment_image(image, image_size):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError("Could not load image")
   
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, image_size)
    img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)
    prediction = model.predict(img_normalized)
    predicted_mask = np.argmax(prediction, axis=-1)[0]
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
    
    # Create RGB mask for visualization
    rgb_mask = np.zeros((*predicted_mask_resized.shape, 3), dtype=np.uint8)
    for class_idx, rgb in class_rgb_values.items():
        rgb_mask[predicted_mask_resized == class_idx] = rgb
    
    return Image.fromarray(cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)), predicted_mask_resized


def create_error_map_overlay(true_mask, predicted_mask, original_image, alpha=0.5):
    """
    Create error map overlay for multi-class segmentation.
    - Green: Correctly predicted (all non-background classes)
    - Red: False Negatives (missed or wrong class)
    - Blue: False Positives (predicted class where should be background)
    - No overlay: True Negatives (correctly predicted background)
    """
    true_mask_array = np.array(true_mask)
    
    # Convert RGB mask to class indices
    if len(true_mask_array.shape) == 3:
        # CRITICAL FIX: Convert true mask from RGB to BGR to match class_rgb_values
        true_mask_bgr = cv2.cvtColor(true_mask_array, cv2.COLOR_RGB2BGR)
        
        true_mask_indices = np.zeros(true_mask_array.shape[:2], dtype=np.uint8)
        for class_idx, bgr in class_rgb_values.items():
            # Now comparing BGR with BGR
            mask = np.all(true_mask_bgr == bgr, axis=-1)
            true_mask_indices[mask] = class_idx
            
            # Debug: print how many pixels found for each class
            if np.sum(mask) > 0:
                print(f"  Class {class_idx}: {np.sum(mask)} pixels")
    else:
        true_mask_indices = true_mask_array
    
    # Correct predictions
    correct_mask = (true_mask_indices == predicted_mask)
    tp_mask = correct_mask & (true_mask_indices > 0)  # Correct non-background
    tn_mask = correct_mask & (true_mask_indices == 0)  # Correct background
    
    # Wrong predictions
    wrong_mask = (true_mask_indices != predicted_mask)
    
    # False Negatives: true mask has a class, but prediction is wrong
    fn_mask = wrong_mask & (true_mask_indices > 0)
    
    # False Positives: prediction has a class, but true mask is background
    fp_mask = wrong_mask & (predicted_mask > 0) & (true_mask_indices == 0)
    
    # Debug: Print counts and percentages
    total_pixels = true_mask_indices.size
    print(f"True Positives (Correct): {np.sum(tp_mask)} pixels ({np.sum(tp_mask)/total_pixels*100:.2f}%)")
    print(f"False Negatives (Missed/Wrong class): {np.sum(fn_mask)} pixels ({np.sum(fn_mask)/total_pixels*100:.2f}%)")
    print(f"False Positives (Background predicted as class): {np.sum(fp_mask)} pixels ({np.sum(fp_mask)/total_pixels*100:.2f}%)")
    print(f"True Negatives (Correct background): {np.sum(tn_mask)} pixels ({np.sum(tn_mask)/total_pixels*100:.2f}%)")
    print(f"Overall Accuracy: {(np.sum(tp_mask) + np.sum(tn_mask))/total_pixels*100:.2f}%")
    
    # Create error map - RGB format for display
    error_map = np.zeros((*true_mask_indices.shape, 3), dtype=np.uint8)
    error_map[tp_mask] = [0, 255, 0]    # TP: Green in RGB
    error_map[fn_mask] = [255, 0, 0]    # FN: Red in RGB
    error_map[fp_mask] = [0, 0, 255]    # FP: Blue in RGB
    # TN remains black (no overlay for background)
    
    # Create overlay - start with original image
    original_array = np.array(original_image)
    if len(original_array.shape) == 2:
        original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
    
    overlay = original_array.copy()
    
    # Overlay all non-background areas (TP, FN, FP) - exclude TN
    overlay_mask = tp_mask | fn_mask | fp_mask
    if np.sum(overlay_mask) > 0:  # Only blend if there are areas to overlay
        overlay[overlay_mask] = (alpha * error_map[overlay_mask] + 
                                  (1 - alpha) * original_array[overlay_mask]).astype(np.uint8)
    
    return Image.fromarray(overlay)



def save_plot_in_vector_format(fig, save_path, formats=['eps', 'png', 'pdf']):
    """
    Save the plot in multiple formats (e.g., EPS, PNG, and PDF).
    Text will be editable in PDF when opened in Illustrator.
    
    Args:
        fig: The Matplotlib figure object to save.
        save_path: Path (including filename without extension) where the plot will be saved.
        formats: List of formats to save the plot (e.g., ['eps', 'png', 'pdf']). 
                 Default is ['eps', 'png', 'pdf'].
    """
    for fmt in formats:
        if fmt not in ['eps', 'svg', 'png', 'pdf']:
            raise ValueError(f"Unsupported format: {fmt}. Use 'eps', 'svg', 'png', or 'pdf'.")
        
        full_path = f"{save_path}.{fmt}"
        
        if fmt == 'pdf':
            # For PDF with editable text
            fig.savefig(full_path, format='pdf', bbox_inches='tight', 
                       dpi=600, metadata={'Creator': 'Matplotlib'})
        elif fmt == 'eps':
            # For EPS with editable text
            fig.savefig(full_path, format='eps', bbox_inches='tight', 
                       dpi=600)
        else:  # For PNG
            fig.savefig(full_path, format=fmt, bbox_inches='tight', dpi=600)
        
        print(f"Plot saved in {fmt.upper()} format at: {full_path}")

def visualize_all_four_plots(images_folder, masks_folder, save_dir, image_size, alpha=0.5):
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No images found in the images folder.")
        return

    for image_file in image_files:
        print(f"Processing image: {image_file}")

        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, image_file)

        try:
            image = Image.open(image_path)
            true_mask = Image.open(mask_path)
            predicted_mask_img, predicted_mask_array = segment_image(image, image_size)
            overlay_image = create_error_map_overlay(true_mask, predicted_mask_array, image, alpha)

            # Visualize four plots side by side
            fig = plt.figure(figsize=(20, 5))

            # Original Image
            plt.subplot(1, 4, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            # True Mask
            plt.subplot(1, 4, 2)
            plt.title("True Mask")
            plt.imshow(true_mask, cmap='cividis')
            plt.axis('off')

            # Predicted Mask
            plt.subplot(1, 4, 3)
            plt.title("Predicted Mask")
            plt.imshow(predicted_mask_img, cmap='cividis')
            plt.axis('off')

            # Error Map Overlay
            plt.subplot(1, 4, 4)
            plt.title(f"Error Map Overlay (alpha={alpha})")
            plt.imshow(overlay_image)
            plt.axis('off')
            plt.show()

            plt.tight_layout()

            # Save the plot with the same name as the image in EPS, PNG, and PDF formats
            image_name = os.path.splitext(image_file)[0]
            save_path = os.path.join(save_dir, image_name)
            save_plot_in_vector_format(fig, save_path, formats=['png', 'pdf'])

            plt.close(fig)  # Close the figure to free memory

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

# Example usage
images_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/external/images"  # Replace with your images folder path
masks_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/external/masks"  # Replace with your masks folder path
save_dir = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET_Small_512_188_epoch/external"

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Adjust alpha value here (0.0 to 1.0)
visualize_all_four_plots(images_folder, masks_folder, save_dir,image_size, alpha=0.5)














































#----------- Prediction with Error Map Entire Folder  Final with Editable PDF Text ----------------------------

import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import normalize
import tensorflow as tf
from keras.saving import register_keras_serializable
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

image_size = (256,256)

# Configure matplotlib for editable text in PDF
matplotlib.rcParams['pdf.fonttype'] = 42  # 42 = TrueType fonts (editable)
matplotlib.rcParams['ps.fonttype'] = 42   # For PostScript (EPS) as well
matplotlib.rcParams['font.family'] = 'sans-serif'  # Use standard fonts
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']  # Common editable fonts

# [Previous custom loss and metric functions remain the same]
# ... (keeping your existing custom functions unchanged)

# Load the model
model_path = r"I:/Radiotherapy/Cervix/models/models/Shahrukh/axial-20260420T031615Z-3-001/axial/Cervix_small_Axial_200_epochs_unet3+.keras"
custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}
model = load_model(model_path, custom_objects=custom_objects, compile=True)

# Class RGB values
class_rgb_values = {
    0: (0, 0, 0),          # Background (black)
    1: (0, 255, 0),        # BODY
    2: (0, 255, 255),      # URINARY BLADDER
    3: (153, 146, 255),      # SMALL BOWEL
    4: (64, 64, 128),      # RECTUM
    5: (255, 255, 0),      # FEMORAL HEAD
    6: (255, 60, 255),     # GTV
    7: (255, 55, 55),      # CTV
}

def segment_image(image, image_size):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError("Could not load image")
   
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, image_size)
    img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)
    prediction = model.predict(img_normalized)
    predicted_mask = np.argmax(prediction, axis=-1)[0]
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
    
    # Create RGB mask for visualization
    rgb_mask = np.zeros((*predicted_mask_resized.shape, 3), dtype=np.uint8)
    for class_idx, rgb in class_rgb_values.items():
        rgb_mask[predicted_mask_resized == class_idx] = rgb
    
    return Image.fromarray(cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)), predicted_mask_resized


def create_error_map_overlay(true_mask, predicted_mask, original_image, alpha=0.5):
    """
    Create error map overlay for multi-class segmentation.
    - Green: Correctly predicted (all non-background classes)
    - Red: False Negatives (missed or wrong class)
    - Blue: False Positives (predicted class where should be background)
    - No overlay: True Negatives (correctly predicted background)
    """
    true_mask_array = np.array(true_mask)
    
    # Convert RGB mask to class indices
    if len(true_mask_array.shape) == 3:
        # CRITICAL FIX: Convert true mask from RGB to BGR to match class_rgb_values
        true_mask_bgr = cv2.cvtColor(true_mask_array, cv2.COLOR_RGB2BGR)
        
        true_mask_indices = np.zeros(true_mask_array.shape[:2], dtype=np.uint8)
        for class_idx, bgr in class_rgb_values.items():
            # Now comparing BGR with BGR
            mask = np.all(true_mask_bgr == bgr, axis=-1)
            true_mask_indices[mask] = class_idx
            
            # Debug: print how many pixels found for each class
            if np.sum(mask) > 0:
                print(f"  Class {class_idx}: {np.sum(mask)} pixels")
    else:
        true_mask_indices = true_mask_array
    
    # Correct predictions
    correct_mask = (true_mask_indices == predicted_mask)
    tp_mask = correct_mask & (true_mask_indices > 0)  # Correct non-background
    tn_mask = correct_mask & (true_mask_indices == 0)  # Correct background
    
    # Wrong predictions
    wrong_mask = (true_mask_indices != predicted_mask)
    
    # False Negatives: true mask has a class, but prediction is wrong
    fn_mask = wrong_mask & (true_mask_indices > 0)
    
    # False Positives: prediction has a class, but true mask is background
    fp_mask = wrong_mask & (predicted_mask > 0) & (true_mask_indices == 0)
    
    # Debug: Print counts and percentages
    total_pixels = true_mask_indices.size
    print(f"True Positives (Correct): {np.sum(tp_mask)} pixels ({np.sum(tp_mask)/total_pixels*100:.2f}%)")
    print(f"False Negatives (Missed/Wrong class): {np.sum(fn_mask)} pixels ({np.sum(fn_mask)/total_pixels*100:.2f}%)")
    print(f"False Positives (Background predicted as class): {np.sum(fp_mask)} pixels ({np.sum(fp_mask)/total_pixels*100:.2f}%)")
    print(f"True Negatives (Correct background): {np.sum(tn_mask)} pixels ({np.sum(tn_mask)/total_pixels*100:.2f}%)")
    print(f"Overall Accuracy: {(np.sum(tp_mask) + np.sum(tn_mask))/total_pixels*100:.2f}%")
    
    # Create error map - RGB format for display
    error_map = np.zeros((*true_mask_indices.shape, 3), dtype=np.uint8)
    error_map[tp_mask] = [0, 255, 0]    # TP: Green in RGB
    error_map[fn_mask] = [255, 0, 0]    # FN: Red in RGB
    error_map[fp_mask] = [0, 0, 255]    # FP: Blue in RGB
    # TN remains black (no overlay for background)
    
    # Create overlay - start with original image
    original_array = np.array(original_image)
    if len(original_array.shape) == 2:
        original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
    
    overlay = original_array.copy()
    
    # Overlay all non-background areas (TP, FN, FP) - exclude TN
    overlay_mask = tp_mask | fn_mask | fp_mask
    if np.sum(overlay_mask) > 0:  # Only blend if there are areas to overlay
        overlay[overlay_mask] = (alpha * error_map[overlay_mask] + 
                                  (1 - alpha) * original_array[overlay_mask]).astype(np.uint8)
    
    return Image.fromarray(overlay)



def save_plot_in_vector_format(fig, save_path, formats=['eps', 'png', 'pdf']):
    """
    Save the plot in multiple formats (e.g., EPS, PNG, and PDF).
    Text will be editable in PDF when opened in Illustrator.
    
    Args:
        fig: The Matplotlib figure object to save.
        save_path: Path (including filename without extension) where the plot will be saved.
        formats: List of formats to save the plot (e.g., ['eps', 'png', 'pdf']). 
                 Default is ['eps', 'png', 'pdf'].
    """
    for fmt in formats:
        if fmt not in ['eps', 'svg', 'png', 'pdf']:
            raise ValueError(f"Unsupported format: {fmt}. Use 'eps', 'svg', 'png', or 'pdf'.")
        
        full_path = f"{save_path}.{fmt}"
        
        if fmt == 'pdf':
            # For PDF with editable text
            fig.savefig(full_path, format='pdf', bbox_inches='tight', 
                       dpi=600, metadata={'Creator': 'Matplotlib'})
        elif fmt == 'eps':
            # For EPS with editable text
            fig.savefig(full_path, format='eps', bbox_inches='tight', 
                       dpi=600)
        else:  # For PNG
            fig.savefig(full_path, format=fmt, bbox_inches='tight', dpi=600)
        
        print(f"Plot saved in {fmt.upper()} format at: {full_path}")

def visualize_all_four_plots(images_folder, masks_folder, save_dir, image_size, alpha=0.5):
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No images found in the images folder.")
        return

    for image_file in image_files:
        print(f"Processing image: {image_file}")

        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, image_file)

        try:
            image = Image.open(image_path)
            true_mask = Image.open(mask_path)
            predicted_mask_img, predicted_mask_array = segment_image(image, image_size)
            overlay_image = create_error_map_overlay(true_mask, predicted_mask_array, image, alpha)

            # Visualize four plots side by side
            fig = plt.figure(figsize=(20, 5))

            # Original Image
            plt.subplot(1, 4, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            # True Mask
            plt.subplot(1, 4, 2)
            plt.title("True Mask")
            plt.imshow(true_mask, cmap='cividis')
            plt.axis('off')

            # Predicted Mask
            plt.subplot(1, 4, 3)
            plt.title("Predicted Mask")
            plt.imshow(predicted_mask_img, cmap='cividis')
            plt.axis('off')

            # Error Map Overlay
            plt.subplot(1, 4, 4)
            plt.title(f"Error Map Overlay (alpha={alpha})")
            plt.imshow(overlay_image)
            plt.axis('off')
            plt.show()

            plt.tight_layout()

            # Save the plot with the same name as the image in EPS, PNG, and PDF formats
            image_name = os.path.splitext(image_file)[0]
            save_path = os.path.join(save_dir, image_name)
            save_plot_in_vector_format(fig, save_path, formats=['png', 'pdf'])

            plt.close(fig)  # Close the figure to free memory

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

# Example usage
images_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/external/images"  # Replace with your images folder path
masks_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/external/masks"  # Replace with your masks folder path
save_dir = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET3+/external"

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Adjust alpha value here (0.0 to 1.0)
visualize_all_four_plots(images_folder, masks_folder, save_dir,image_size, alpha=0.5)

















































#----------- Prediction with Error Map Entire Folder  Final with Editable PDF Text ----------------------------

import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import normalize
import tensorflow as tf
from keras.saving import register_keras_serializable
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

image_size = (256,256)

# Configure matplotlib for editable text in PDF
matplotlib.rcParams['pdf.fonttype'] = 42  # 42 = TrueType fonts (editable)
matplotlib.rcParams['ps.fonttype'] = 42   # For PostScript (EPS) as well
matplotlib.rcParams['font.family'] = 'sans-serif'  # Use standard fonts
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']  # Common editable fonts

# [Previous custom loss and metric functions remain the same]
# ... (keeping your existing custom functions unchanged)

# Load the model
model_path = r"I:/Radiotherapy/Cervix/models/models/Shahrukh/axial-20260420T031615Z-3-001/axial/Cervix_small_axial_epoch100_best_unet_base.keras"
custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}
model = load_model(model_path, custom_objects=custom_objects, compile=True)

# Class RGB values
class_rgb_values = {
    0: (0, 0, 0),          # Background (black)
    1: (0, 255, 0),        # BODY
    2: (0, 255, 255),      # URINARY BLADDER
    3: (153, 146, 255),      # SMALL BOWEL
    4: (64, 64, 128),      # RECTUM
    5: (255, 255, 0),      # FEMORAL HEAD
    6: (255, 60, 255),     # GTV
    7: (255, 55, 55),      # CTV
}

def segment_image(image, image_size):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError("Could not load image")
   
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, image_size)
    img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)
    prediction = model.predict(img_normalized)
    predicted_mask = np.argmax(prediction, axis=-1)[0]
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
    
    # Create RGB mask for visualization
    rgb_mask = np.zeros((*predicted_mask_resized.shape, 3), dtype=np.uint8)
    for class_idx, rgb in class_rgb_values.items():
        rgb_mask[predicted_mask_resized == class_idx] = rgb
    
    return Image.fromarray(cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)), predicted_mask_resized


def create_error_map_overlay(true_mask, predicted_mask, original_image, alpha=0.5):
    """
    Create error map overlay for multi-class segmentation.
    - Green: Correctly predicted (all non-background classes)
    - Red: False Negatives (missed or wrong class)
    - Blue: False Positives (predicted class where should be background)
    - No overlay: True Negatives (correctly predicted background)
    """
    true_mask_array = np.array(true_mask)
    
    # Convert RGB mask to class indices
    if len(true_mask_array.shape) == 3:
        # CRITICAL FIX: Convert true mask from RGB to BGR to match class_rgb_values
        true_mask_bgr = cv2.cvtColor(true_mask_array, cv2.COLOR_RGB2BGR)
        
        true_mask_indices = np.zeros(true_mask_array.shape[:2], dtype=np.uint8)
        for class_idx, bgr in class_rgb_values.items():
            # Now comparing BGR with BGR
            mask = np.all(true_mask_bgr == bgr, axis=-1)
            true_mask_indices[mask] = class_idx
            
            # Debug: print how many pixels found for each class
            if np.sum(mask) > 0:
                print(f"  Class {class_idx}: {np.sum(mask)} pixels")
    else:
        true_mask_indices = true_mask_array
    
    # Correct predictions
    correct_mask = (true_mask_indices == predicted_mask)
    tp_mask = correct_mask & (true_mask_indices > 0)  # Correct non-background
    tn_mask = correct_mask & (true_mask_indices == 0)  # Correct background
    
    # Wrong predictions
    wrong_mask = (true_mask_indices != predicted_mask)
    
    # False Negatives: true mask has a class, but prediction is wrong
    fn_mask = wrong_mask & (true_mask_indices > 0)
    
    # False Positives: prediction has a class, but true mask is background
    fp_mask = wrong_mask & (predicted_mask > 0) & (true_mask_indices == 0)
    
    # Debug: Print counts and percentages
    total_pixels = true_mask_indices.size
    print(f"True Positives (Correct): {np.sum(tp_mask)} pixels ({np.sum(tp_mask)/total_pixels*100:.2f}%)")
    print(f"False Negatives (Missed/Wrong class): {np.sum(fn_mask)} pixels ({np.sum(fn_mask)/total_pixels*100:.2f}%)")
    print(f"False Positives (Background predicted as class): {np.sum(fp_mask)} pixels ({np.sum(fp_mask)/total_pixels*100:.2f}%)")
    print(f"True Negatives (Correct background): {np.sum(tn_mask)} pixels ({np.sum(tn_mask)/total_pixels*100:.2f}%)")
    print(f"Overall Accuracy: {(np.sum(tp_mask) + np.sum(tn_mask))/total_pixels*100:.2f}%")
    
    # Create error map - RGB format for display
    error_map = np.zeros((*true_mask_indices.shape, 3), dtype=np.uint8)
    error_map[tp_mask] = [0, 255, 0]    # TP: Green in RGB
    error_map[fn_mask] = [255, 0, 0]    # FN: Red in RGB
    error_map[fp_mask] = [0, 0, 255]    # FP: Blue in RGB
    # TN remains black (no overlay for background)
    
    # Create overlay - start with original image
    original_array = np.array(original_image)
    if len(original_array.shape) == 2:
        original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
    
    overlay = original_array.copy()
    
    # Overlay all non-background areas (TP, FN, FP) - exclude TN
    overlay_mask = tp_mask | fn_mask | fp_mask
    if np.sum(overlay_mask) > 0:  # Only blend if there are areas to overlay
        overlay[overlay_mask] = (alpha * error_map[overlay_mask] + 
                                  (1 - alpha) * original_array[overlay_mask]).astype(np.uint8)
    
    return Image.fromarray(overlay)



def save_plot_in_vector_format(fig, save_path, formats=['eps', 'png', 'pdf']):
    """
    Save the plot in multiple formats (e.g., EPS, PNG, and PDF).
    Text will be editable in PDF when opened in Illustrator.
    
    Args:
        fig: The Matplotlib figure object to save.
        save_path: Path (including filename without extension) where the plot will be saved.
        formats: List of formats to save the plot (e.g., ['eps', 'png', 'pdf']). 
                 Default is ['eps', 'png', 'pdf'].
    """
    for fmt in formats:
        if fmt not in ['eps', 'svg', 'png', 'pdf']:
            raise ValueError(f"Unsupported format: {fmt}. Use 'eps', 'svg', 'png', or 'pdf'.")
        
        full_path = f"{save_path}.{fmt}"
        
        if fmt == 'pdf':
            # For PDF with editable text
            fig.savefig(full_path, format='pdf', bbox_inches='tight', 
                       dpi=600, metadata={'Creator': 'Matplotlib'})
        elif fmt == 'eps':
            # For EPS with editable text
            fig.savefig(full_path, format='eps', bbox_inches='tight', 
                       dpi=600)
        else:  # For PNG
            fig.savefig(full_path, format=fmt, bbox_inches='tight', dpi=600)
        
        print(f"Plot saved in {fmt.upper()} format at: {full_path}")

def visualize_all_four_plots(images_folder, masks_folder, save_dir, image_size, alpha=0.5):
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No images found in the images folder.")
        return

    for image_file in image_files:
        print(f"Processing image: {image_file}")

        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, image_file)

        try:
            image = Image.open(image_path)
            true_mask = Image.open(mask_path)
            predicted_mask_img, predicted_mask_array = segment_image(image, image_size)
            overlay_image = create_error_map_overlay(true_mask, predicted_mask_array, image, alpha)

            # Visualize four plots side by side
            fig = plt.figure(figsize=(20, 5))

            # Original Image
            plt.subplot(1, 4, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            # True Mask
            plt.subplot(1, 4, 2)
            plt.title("True Mask")
            plt.imshow(true_mask, cmap='cividis')
            plt.axis('off')

            # Predicted Mask
            plt.subplot(1, 4, 3)
            plt.title("Predicted Mask")
            plt.imshow(predicted_mask_img, cmap='cividis')
            plt.axis('off')

            # Error Map Overlay
            plt.subplot(1, 4, 4)
            plt.title(f"Error Map Overlay (alpha={alpha})")
            plt.imshow(overlay_image)
            plt.axis('off')
            plt.show()

            plt.tight_layout()

            # Save the plot with the same name as the image in EPS, PNG, and PDF formats
            image_name = os.path.splitext(image_file)[0]
            save_path = os.path.join(save_dir, image_name)
            save_plot_in_vector_format(fig, save_path, formats=['png', 'pdf'])

            plt.close(fig)  # Close the figure to free memory

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

# Example usage
images_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/external/images"  # Replace with your images folder path
masks_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/external/masks"  # Replace with your masks folder path
save_dir = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET_base/external"

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Adjust alpha value here (0.0 to 1.0)
visualize_all_four_plots(images_folder, masks_folder, save_dir,image_size, alpha=0.5)















































#----------- Prediction with Error Map Entire Folder  Final with Editable PDF Text ----------------------------

import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import normalize
import tensorflow as tf
from keras.saving import register_keras_serializable
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

image_size = (256,256)

# Configure matplotlib for editable text in PDF
matplotlib.rcParams['pdf.fonttype'] = 42  # 42 = TrueType fonts (editable)
matplotlib.rcParams['ps.fonttype'] = 42   # For PostScript (EPS) as well
matplotlib.rcParams['font.family'] = 'sans-serif'  # Use standard fonts
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']  # Common editable fonts

# [Previous custom loss and metric functions remain the same]
# ... (keeping your existing custom functions unchanged)

# Load the model
model_path = r"I:/Radiotherapy/Cervix/models/models/Shahrukh/axial-20260420T031615Z-3-001/axial/Cervix_small_Axial_200_epochs_vanilla_unet.keras"
custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}
model = load_model(model_path, custom_objects=custom_objects, compile=True)

# Class RGB values
class_rgb_values = {
    0: (0, 0, 0),          # Background (black)
    1: (0, 255, 0),        # BODY
    2: (0, 255, 255),      # URINARY BLADDER
    3: (153, 146, 255),      # SMALL BOWEL
    4: (64, 64, 128),      # RECTUM
    5: (255, 255, 0),      # FEMORAL HEAD
    6: (255, 60, 255),     # GTV
    7: (255, 55, 55),      # CTV
}

def segment_image(image, image_size):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError("Could not load image")
   
    original_shape = img.shape[:2]
    img_resized = cv2.resize(img, image_size)
    img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)
    prediction = model.predict(img_normalized)
    predicted_mask = np.argmax(prediction, axis=-1)[0]
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
    
    # Create RGB mask for visualization
    rgb_mask = np.zeros((*predicted_mask_resized.shape, 3), dtype=np.uint8)
    for class_idx, rgb in class_rgb_values.items():
        rgb_mask[predicted_mask_resized == class_idx] = rgb
    
    return Image.fromarray(cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)), predicted_mask_resized


def create_error_map_overlay(true_mask, predicted_mask, original_image, alpha=0.5):
    """
    Create error map overlay for multi-class segmentation.
    - Green: Correctly predicted (all non-background classes)
    - Red: False Negatives (missed or wrong class)
    - Blue: False Positives (predicted class where should be background)
    - No overlay: True Negatives (correctly predicted background)
    """
    true_mask_array = np.array(true_mask)
    
    # Convert RGB mask to class indices
    if len(true_mask_array.shape) == 3:
        # CRITICAL FIX: Convert true mask from RGB to BGR to match class_rgb_values
        true_mask_bgr = cv2.cvtColor(true_mask_array, cv2.COLOR_RGB2BGR)
        
        true_mask_indices = np.zeros(true_mask_array.shape[:2], dtype=np.uint8)
        for class_idx, bgr in class_rgb_values.items():
            # Now comparing BGR with BGR
            mask = np.all(true_mask_bgr == bgr, axis=-1)
            true_mask_indices[mask] = class_idx
            
            # Debug: print how many pixels found for each class
            if np.sum(mask) > 0:
                print(f"  Class {class_idx}: {np.sum(mask)} pixels")
    else:
        true_mask_indices = true_mask_array
    
    # Correct predictions
    correct_mask = (true_mask_indices == predicted_mask)
    tp_mask = correct_mask & (true_mask_indices > 0)  # Correct non-background
    tn_mask = correct_mask & (true_mask_indices == 0)  # Correct background
    
    # Wrong predictions
    wrong_mask = (true_mask_indices != predicted_mask)
    
    # False Negatives: true mask has a class, but prediction is wrong
    fn_mask = wrong_mask & (true_mask_indices > 0)
    
    # False Positives: prediction has a class, but true mask is background
    fp_mask = wrong_mask & (predicted_mask > 0) & (true_mask_indices == 0)
    
    # Debug: Print counts and percentages
    total_pixels = true_mask_indices.size
    print(f"True Positives (Correct): {np.sum(tp_mask)} pixels ({np.sum(tp_mask)/total_pixels*100:.2f}%)")
    print(f"False Negatives (Missed/Wrong class): {np.sum(fn_mask)} pixels ({np.sum(fn_mask)/total_pixels*100:.2f}%)")
    print(f"False Positives (Background predicted as class): {np.sum(fp_mask)} pixels ({np.sum(fp_mask)/total_pixels*100:.2f}%)")
    print(f"True Negatives (Correct background): {np.sum(tn_mask)} pixels ({np.sum(tn_mask)/total_pixels*100:.2f}%)")
    print(f"Overall Accuracy: {(np.sum(tp_mask) + np.sum(tn_mask))/total_pixels*100:.2f}%")
    
    # Create error map - RGB format for display
    error_map = np.zeros((*true_mask_indices.shape, 3), dtype=np.uint8)
    error_map[tp_mask] = [0, 255, 0]    # TP: Green in RGB
    error_map[fn_mask] = [255, 0, 0]    # FN: Red in RGB
    error_map[fp_mask] = [0, 0, 255]    # FP: Blue in RGB
    # TN remains black (no overlay for background)
    
    # Create overlay - start with original image
    original_array = np.array(original_image)
    if len(original_array.shape) == 2:
        original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
    
    overlay = original_array.copy()
    
    # Overlay all non-background areas (TP, FN, FP) - exclude TN
    overlay_mask = tp_mask | fn_mask | fp_mask
    if np.sum(overlay_mask) > 0:  # Only blend if there are areas to overlay
        overlay[overlay_mask] = (alpha * error_map[overlay_mask] + 
                                  (1 - alpha) * original_array[overlay_mask]).astype(np.uint8)
    
    return Image.fromarray(overlay)



def save_plot_in_vector_format(fig, save_path, formats=['eps', 'png', 'pdf']):
    """
    Save the plot in multiple formats (e.g., EPS, PNG, and PDF).
    Text will be editable in PDF when opened in Illustrator.
    
    Args:
        fig: The Matplotlib figure object to save.
        save_path: Path (including filename without extension) where the plot will be saved.
        formats: List of formats to save the plot (e.g., ['eps', 'png', 'pdf']). 
                 Default is ['eps', 'png', 'pdf'].
    """
    for fmt in formats:
        if fmt not in ['eps', 'svg', 'png', 'pdf']:
            raise ValueError(f"Unsupported format: {fmt}. Use 'eps', 'svg', 'png', or 'pdf'.")
        
        full_path = f"{save_path}.{fmt}"
        
        if fmt == 'pdf':
            # For PDF with editable text
            fig.savefig(full_path, format='pdf', bbox_inches='tight', 
                       dpi=600, metadata={'Creator': 'Matplotlib'})
        elif fmt == 'eps':
            # For EPS with editable text
            fig.savefig(full_path, format='eps', bbox_inches='tight', 
                       dpi=600)
        else:  # For PNG
            fig.savefig(full_path, format=fmt, bbox_inches='tight', dpi=600)
        
        print(f"Plot saved in {fmt.upper()} format at: {full_path}")

def visualize_all_four_plots(images_folder, masks_folder, save_dir, image_size, alpha=0.5):
    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not image_files:
        print("No images found in the images folder.")
        return

    for image_file in image_files:
        print(f"Processing image: {image_file}")

        image_path = os.path.join(images_folder, image_file)
        mask_path = os.path.join(masks_folder, image_file)

        try:
            image = Image.open(image_path)
            true_mask = Image.open(mask_path)
            predicted_mask_img, predicted_mask_array = segment_image(image, image_size)
            overlay_image = create_error_map_overlay(true_mask, predicted_mask_array, image, alpha)

            # Visualize four plots side by side
            fig = plt.figure(figsize=(20, 5))

            # Original Image
            plt.subplot(1, 4, 1)
            plt.title("Original Image")
            plt.imshow(image, cmap='gray')
            plt.axis('off')

            # True Mask
            plt.subplot(1, 4, 2)
            plt.title("True Mask")
            plt.imshow(true_mask, cmap='cividis')
            plt.axis('off')

            # Predicted Mask
            plt.subplot(1, 4, 3)
            plt.title("Predicted Mask")
            plt.imshow(predicted_mask_img, cmap='cividis')
            plt.axis('off')

            # Error Map Overlay
            plt.subplot(1, 4, 4)
            plt.title(f"Error Map Overlay (alpha={alpha})")
            plt.imshow(overlay_image)
            plt.axis('off')
            plt.show()

            plt.tight_layout()

            # Save the plot with the same name as the image in EPS, PNG, and PDF formats
            image_name = os.path.splitext(image_file)[0]
            save_path = os.path.join(save_dir, image_name)
            save_plot_in_vector_format(fig, save_path, formats=['png', 'pdf'])

            plt.close(fig)  # Close the figure to free memory

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

# Example usage
images_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/external/images"  # Replace with your images folder path
masks_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/external/masks"  # Replace with your masks folder path
save_dir = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET/external"

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Adjust alpha value here (0.0 to 1.0)
visualize_all_four_plots(images_folder, masks_folder, save_dir,image_size, alpha=0.5)






















































# Cell 4: Multi-Model Error Analysis with Per-Model Output Organization
# =============================================================================
# Description: 
#   Loads multiple segmentation models and generates prediction visualizations
#   (Original, Ground Truth, Predicted Mask, Error Map) for each model.
#   Results are automatically organized into separate folders per model.
#
# Features:
#   - Supports any number of models (1, 3, 5, etc.) dynamically
#   - Creates dedicated folders for each model based on provided name
#   - Saves figures in PNG, EPS, and PDF formats
#   - Single tqdm progress bar tracking total tasks (images × models)
#   - Automatic error map overlay generation
#
# Usage:
#   1. Configure model paths and names in model_configs below
#   2. Set 'enabled': True for models you want to test
#   3. Specify input/output folder paths
#   4. Run the cell
# =============================================================================

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import normalize
from tqdm import tqdm

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 6

# Check required dependencies
if 'create_error_map_overlay' not in globals():
    raise NameError("create_error_map_overlay is not defined. Run the earlier helper cell first.")

if 'class_rgb_values' not in globals():
    raise NameError("class_rgb_values is not defined. Run the earlier helper cell first.")

def load_multiple_models(model_configs, custom_objects=None, max_models=5):
    """Load multiple models based on configuration dictionary."""
    loaded_models = []

    if isinstance(model_configs, dict):
        model_configs = [model_configs]

    enabled_configs = [cfg for cfg in model_configs if cfg.get('enabled', False)]

    if len(enabled_configs) > max_models:
        print(f"More than {max_models} enabled models found. Using the first {max_models}.")
        enabled_configs = enabled_configs[:max_models]

    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    
    for cfg in enabled_configs:
        model_name = cfg.get('name', 'Unnamed Model')
        model_path = cfg.get('path', '')
        image_size = tuple(cfg.get('image_size', (256, 256)))

        if not model_path:
            print(f"\n⚠ Skipping {model_name}: empty model path.")
            continue

        if not os.path.exists(model_path):
            print(f"\n⚠ Skipping {model_name}: model path not found -> {model_path}")
            continue

        try:
            model = load_model(model_path, custom_objects=custom_objects, compile=True)
            loaded_models.append({
                'name': model_name,
                'model': model,
                'image_size': image_size,
            })
            print(f"\n✓ Loaded {model_name}")
        except Exception as exc:
            print(f"\n✗ Failed to load {model_name}: {exc}")

    print("\n" + "="*60)
    print(f"✓ Total loaded models: {len(loaded_models)}")
    if loaded_models:
        print("✓ Loaded model names:", ', '.join(item['name'] for item in loaded_models))
    print("="*60 + "\n")

    return loaded_models

def segment_with_loaded_model(model, image, image_size):
    """Perform segmentation using a loaded model."""
    image_rgb = np.array(image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    original_height, original_width = image_bgr.shape[:2]
    resized = cv2.resize(image_bgr, image_size)
    normalized = normalize(np.array([resized], dtype=np.float32), axis=1)

    prediction = model.predict(normalized, verbose=0)
    pred_mask = np.argmax(prediction, axis=-1)[0]
    pred_mask = cv2.resize(pred_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    pred_rgb = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_idx, rgb_value in class_rgb_values.items():
        pred_rgb[pred_mask == class_idx] = rgb_value

    pred_rgb = cv2.cvtColor(pred_rgb, cv2.COLOR_BGR2RGB)
    return Image.fromarray(pred_rgb), pred_mask

def find_matching_mask(masks_folder, image_file):
    """Find corresponding mask file for a given image."""
    image_stem = os.path.splitext(image_file)[0]
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    for ext in valid_exts:
        candidate = os.path.join(masks_folder, image_stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def save_figure_in_formats(fig, save_base_path, formats=('png', 'eps', 'pdf')):
    """Save figure in multiple formats."""
    for fmt in formats:
        output_path = f"{save_base_path}.{fmt}"
        fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight')

def visualize_multi_model_results_per_model(loaded_models, images_folder, masks_folder, save_root_dir, alpha=0.5, max_images=None):
    """
    Saves results in separate folders per model (dynamically based on loaded_models)
    
    Directory structure:
        save_root_dir/
            Model_A/
                image1.png, image1.eps, image1.pdf
                image2.png, image2.eps, image2.pdf
            Model_B/
                ...
            Model_C/
                ...
    """
    if not loaded_models:
        print("❌ No models were loaded. Check the model paths and enabled flags.")
        return

    if not os.path.exists(images_folder):
        print(f"❌ Images folder not found: {images_folder}")
        return

    if not os.path.exists(masks_folder):
        print(f"❌ Masks folder not found: {masks_folder}")
        return

    os.makedirs(save_root_dir, exist_ok=True)

    # Get list of image files
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = sorted([f for f in os.listdir(images_folder) if f.lower().endswith(valid_exts)])

    if max_images is not None:
        image_files = image_files[:max_images]

    if not image_files:
        print("❌ No images found to process.")
        return

    # Print processing info
    print("\n" + "="*60)
    print("PROCESSING INFORMATION")
    print("="*60)
    print(f"📁 Images folder: {images_folder}")
    print(f"📁 Masks folder: {masks_folder}")
    print(f"📁 Output root: {save_root_dir}")
    print(f"🖼️  Images to process: {len(image_files)}")
    print(f"🤖 Models loaded: {len(loaded_models)}")
    print(f"📊 Total tasks: {len(image_files) * len(loaded_models)} images to process")
    print(f"🎨 Error map alpha: {alpha}")
    print("="*60 + "\n")

    # Create folders for each model
    print("📁 Creating output folders...")
    for model_info in loaded_models:
        model_name = model_info['name']
        folder_name = model_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        model_save_dir = os.path.join(save_root_dir, folder_name)
        os.makedirs(model_save_dir, exist_ok=True)
        print(f"  ✓ Created: {model_save_dir}")
    print()

    # Pre-filter images with valid masks
    print("🔍 Checking for matching masks...")
    valid_images = []
    for image_file in image_files:
        mask_path = find_matching_mask(masks_folder, image_file)
        if mask_path is not None:
            valid_images.append((image_file, mask_path))
        else:
            print(f"  ⚠ No mask found for: {image_file}")
    
    if not valid_images:
        print("❌ No valid image-mask pairs found.")
        return
    
    print(f"✓ Found {len(valid_images)} valid image-mask pairs\n")
    
    # Calculate total tasks
    total_tasks = len(valid_images) * len(loaded_models)
    
    # Process all combinations with a single progress bar
    print("🔄 Processing all model-image combinations...")
    
    with tqdm(total=total_tasks, desc="Overall progress", unit="task", 
              bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}') as pbar:
        
        for image_file, mask_path in valid_images:
            try:
                image = Image.open(os.path.join(images_folder, image_file)).convert('RGB')
                true_mask = Image.open(mask_path)
            except Exception as exc:
                print(f"\n❌ Failed to open {image_file}: {exc}")
                pbar.update(len(loaded_models))  # Skip this image for all models
                continue
            
            # Process each model for this image
            for model_info in loaded_models:
                model_name = model_info['name']
                model = model_info['model']
                image_size = model_info['image_size']
                
                # Get folder name
                folder_name = model_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                model_save_dir = os.path.join(save_root_dir, folder_name)
                
                try:
                    # Get prediction and create visualization
                    pred_image, pred_mask = segment_with_loaded_model(model, image, image_size)
                    error_overlay = create_error_map_overlay(true_mask, pred_mask, image, alpha=alpha)
                    
                    # Create figure
                    fig, axes = plt.subplots(1, 4, figsize=(10, 3.5), squeeze=True)
                    
                    axes[0].imshow(image)
                    axes[0].set_title("Original", fontsize=8)
                    axes[0].axis('off')
                    
                    axes[1].imshow(true_mask, cmap='cividis')
                    axes[1].set_title("Ground Truth", fontsize=8)
                    axes[1].axis('off')
                    
                    axes[2].imshow(pred_image)
                    axes[2].set_title("Predicted Mask", fontsize=8)
                    axes[2].axis('off')
                    
                    axes[3].imshow(error_overlay)
                    axes[3].set_title("Error Map", fontsize=8)
                    axes[3].axis('off')
                    
                    plt.suptitle(f"Model: {model_name}", fontsize=10, y=1.02)
                    plt.tight_layout()
                    
                    # Save figure
                    image_stem = os.path.splitext(image_file)[0]
                    save_base_path = os.path.join(model_save_dir, f"{image_stem}")
                    save_figure_in_formats(fig, save_base_path, formats=('png', 'eps', 'pdf'))
                    
                    plt.close(fig)
                    
                    # Update progress bar with custom message
                    pbar.set_postfix_str(f"{image_file} | {model_name}")
                    pbar.update(1)
                    
                except Exception as exc:
                    print(f"\n❌ Error processing {image_file} with {model_name}: {exc}")
                    pbar.update(1)
                    continue

    # Print completion summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"✓ Processed {len(valid_images)} images")
    print(f"✓ Generated predictions for {len(loaded_models)} models")
    print(f"✓ Total tasks completed: {total_tasks}")
    print(f"✓ Results saved in:")
    for model_info in loaded_models:
        folder_name = model_info['name'].replace(' ', '_').replace('/', '_').replace('\\', '_')
        print(f"  📁 {save_root_dir}/{folder_name}/")
    print("="*60 + "\n")


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Configure your models below. The code will automatically work for ANY number
# of enabled models (1, 3, 5, etc.). Set 'enabled': True for models you want to use.
# =============================================================================

model_configs = [
    {
        'name': 'unet3+',
        'path': r'I:/Radiotherapy/Cervix/models/models/Shahrukh/axial-20260420T031615Z-3-001/axial/Cervix_small_Axial_200_epochs_unet3+.keras',
        'enabled': True,
        'image_size': (256, 256),
    },
    {
        'name': 'vanilla_unet',
        'path': r'I:/Radiotherapy/Cervix/models/models/Shahrukh/axial-20260420T031615Z-3-001/axial/Cervix_small_Axial_200_epochs_vanilla_unet.keras',
        'enabled': True,
        'image_size': (256, 256),
    },
    {
        'name': 'best_unet_base',
        'path': r'I:/Radiotherapy/Cervix/models/models/Shahrukh/axial-20260420T031615Z-3-001/axial/Cervix_small_axial_epoch100_best_unet_base.keras',
        'enabled': True,
        'image_size': (256, 256),
    },
    {
        'name': 'Model D',
        'path': r'',
        'enabled': False,
        'image_size': (256, 256),
    },
    {
        'name': 'Model E',
        'path': r'',
        'enabled': False,
        'image_size': (256, 256),
    },
]

# =============================================================================
# INPUT/OUTPUT PATHS
# =============================================================================
images_folder = r"I:/Radiotherapy/Cervix/cervix_small_set/patient_wise/images/R1811001588"
masks_folder = r"I:/Radiotherapy/Cervix/cervix_small_set/patient_wise/masks/R1811001588"
save_root_dir = r"I:\\Radiotherapy\\Cervix\\Paper\\code\\test_multi_model"

# =============================================================================
# EXECUTION
# =============================================================================
if __name__ == "__main__":
    # Load models (automatically picks only enabled ones)
    custom_objects_for_loading = globals().get('custom_objects', None)
    loaded_models = load_multiple_models(model_configs, custom_objects=custom_objects_for_loading, max_models=5)

    # Run visualization if models were loaded
    if loaded_models:
        visualize_multi_model_results_per_model(
            loaded_models=loaded_models,
            images_folder=images_folder,
            masks_folder=masks_folder,
            save_root_dir=save_root_dir,
            alpha=0.5,
            max_images=None,  # Set to a number (e.g., 10) to limit images
        )
    else:
        print("❌ No models were successfully loaded. Please check your model paths.")





















































        
        
        
        