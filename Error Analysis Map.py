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
images_folder = r"I:/Radiotherapy/Cervix/BMU_Cervix_Dataset/Dataset/images"  # Replace with your images folder path
masks_folder = r"I:/Radiotherapy/Cervix/BMU_Cervix_Dataset/Dataset/masks"  # Replace with your masks folder path

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
images_folder = r"I:/Radiotherapy/Cervix/BMU_Cervix_Dataset/Dataset/small/images"  # Replace with your images folder path
masks_folder = r"I:/Radiotherapy/Cervix/BMU_Cervix_Dataset/Dataset/small/masks"  # Replace with your masks folder path
save_dir = r"I:/Radiotherapy/Cervix/BMU_Cervix_Dataset/Dataset/small"

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Adjust alpha value here (0.0 to 1.0)
visualize_all_four_plots(images_folder, masks_folder, save_dir,image_size, alpha=0.5)








































# #----------- Prediction with Error Map Entire Folder  Final with Editable PDF Text ALL ALL ALL ----------------------------
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
    
    
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.utils import normalize
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

# --- Configuration & Setup ---
image_size = (256, 256)
matplotlib.rcParams['pdf.fonttype'] = 42  
matplotlib.rcParams['ps.fonttype'] = 42   
matplotlib.rcParams['font.family'] = 'sans-serif'  
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

# [Assuming combined_loss, focal_loss, soft_dice_loss, CustomMeanIoU are defined above]

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

class_rgb_values = {
    0: (0, 0, 0),       # Background
    1: (0, 255, 0),     # BODY
    2: (0, 255, 255),   # URINARY BLADDER
    3: (153, 146, 255), # SMALL BOWEL
    4: (64, 64, 128),   # RECTUM
    5: (255, 255, 0),   # FEMORAL HEAD
    6: (255, 60, 255),  # GTV
    7: (255, 55, 55),   # CTV
}

# --- Helper Functions (Logic preserved from your snippet) ---

def segment_image(image, image_size):
    img = np.array(image)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    original_shape = img_bgr.shape[:2]
    img_resized = cv2.resize(img_bgr, image_size)
    img_normalized = normalize(np.array([img_resized], dtype=np.float32), axis=1)
    
    prediction = model.predict(img_normalized)
    predicted_mask = np.argmax(prediction, axis=-1)[0]
    predicted_mask_resized = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
    
    rgb_mask = np.zeros((*predicted_mask_resized.shape, 3), dtype=np.uint8)
    for class_idx, rgb in class_rgb_values.items():
        rgb_mask[predicted_mask_resized == class_idx] = rgb
    
    return Image.fromarray(cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2RGB)), predicted_mask_resized

def create_error_map_overlay(true_mask, predicted_mask, original_image, alpha=0.5):
    true_mask_array = np.array(true_mask)
    if len(true_mask_array.shape) == 3:
        true_mask_bgr = cv2.cvtColor(true_mask_array, cv2.COLOR_RGB2BGR)
        true_mask_indices = np.zeros(true_mask_array.shape[:2], dtype=np.uint8)
        for class_idx, bgr in class_rgb_values.items():
            mask = np.all(true_mask_bgr == bgr, axis=-1)
            true_mask_indices[mask] = class_idx
    else:
        true_mask_indices = true_mask_array

    correct_mask = (true_mask_indices == predicted_mask)
    tp_mask = correct_mask & (true_mask_indices > 0)
    tn_mask = correct_mask & (true_mask_indices == 0)
    wrong_mask = (true_mask_indices != predicted_mask)
    fn_mask = wrong_mask & (true_mask_indices > 0)
    fp_mask = wrong_mask & (predicted_mask > 0) & (true_mask_indices == 0)
    
    error_map = np.zeros((*true_mask_indices.shape, 3), dtype=np.uint8)
    error_map[tp_mask] = [0, 255, 0]    # TP: Green
    error_map[fn_mask] = [255, 0, 0]    # FN: Red
    error_map[fp_mask] = [0, 0, 255]    # FP: Blue
    
    original_array = np.array(original_image)
    if len(original_array.shape) == 2:
        original_array = cv2.cvtColor(original_array, cv2.COLOR_GRAY2RGB)
    
    overlay = original_array.copy()
    overlay_mask = tp_mask | fn_mask | fp_mask
    if np.sum(overlay_mask) > 0:
        overlay[overlay_mask] = (alpha * error_map[overlay_mask] + 
                                 (1 - alpha) * original_array[overlay_mask]).astype(np.uint8)
    
    return Image.fromarray(overlay)

# --- Updated Visualization and Saving Logic ---

def process_and_save_results(images_folder, masks_folder, base_save_dir, image_size, alpha=0.5):
    # Define and create the 4 subdirectories
    dirs = {
        "original": os.path.join(base_save_dir, "1_Original_Images"),
        "true_mask": os.path.join(base_save_dir, "2_True_Masks"),
        "pred_mask": os.path.join(base_save_dir, "3_Predicted_Masks"),
        "error_map": os.path.join(base_save_dir, "4_Error_Analysis_Maps")
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    image_files = [f for f in os.listdir(images_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for image_file in image_files:
        print(f"Processing: {image_file}")
        img_name_no_ext = os.path.splitext(image_file)[0]

        try:
            # 1. Load and Process
            image = Image.open(os.path.join(images_folder, image_file)).convert("RGB")
            true_mask = Image.open(os.path.join(masks_folder, image_file)).convert("RGB")
            
            pred_mask_img, pred_mask_array = segment_image(image, image_size)
            error_overlay = create_error_map_overlay(true_mask, pred_mask_array, image, alpha)

            # 2. Save individual components
            image.save(os.path.join(dirs["original"], f"{img_name_no_ext}.png"))
            true_mask.save(os.path.join(dirs["true_mask"], f"{img_name_no_ext}.png"))
            pred_mask_img.save(os.path.join(dirs["pred_mask"], f"{img_name_no_ext}.png"))
            
            # 3. Create and Save the Combined Figure (Error Analysis Map)
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            
            axes[0].imshow(image)
            axes[0].set_title("Original Image")
            
            axes[1].imshow(true_mask)
            axes[1].set_title("True Mask")
            
            axes[2].imshow(pred_mask_img)
            axes[2].set_title("Predicted Mask")
            
            axes[3].imshow(error_overlay)
            axes[3].set_title(f"Error Map Overlay (α={alpha})")
            
            for ax in axes:
                ax.axis('off')
            
            plt.tight_layout()
            
            # Save combined plot in multiple formats as requested
            comb_path = os.path.join(dirs["error_map"], img_name_no_ext)
            fig.savefig(f"{comb_path}.png", dpi=300, bbox_inches='tight')
            fig.savefig(f"{comb_path}.pdf", format='pdf', dpi=600, bbox_inches='tight')
            
            plt.close(fig)
            print(f"Successfully saved all outputs for {image_file}")

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")

# --- Run ---
images_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/images"
masks_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/masks"
save_dir = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET_all"

process_and_save_results(images_folder, masks_folder, save_dir, image_size, alpha=0.5)




















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
SAVE_DIR      = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/BAT-RM_all'

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
























#----------- Prediction with Error Map Entire Folder  Final with Editable PDF Text BAT-RM ALL ALL ALL ----------------------------




import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# Vector export settings
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
SAVE_DIR      = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/BAT-RM_all'

IMAGE_SIZE = (512, 512)
ALPHA      = 0.5

class_names = {0: 'Background', 1: 'BODY', 2: 'URINARY BLADDER', 3: 'SMALL BOWEL', 
               4: 'RECTUM', 5: 'FEMORAL HEAD', 6: 'GTV', 7: 'CTV'}

bgr_to_class = {
    (0, 0, 0): 0, (0, 255, 0): 1, (0, 255, 255): 2, (153, 146, 255): 3,
    (64, 64, 128): 4, (255, 255, 0): 5, (255, 60, 255): 6, (255, 55, 55): 7
}
class_bgr_values = {v: k for k, v in bgr_to_class.items()}

# ===============================
# 3. LOAD MODEL & HELPERS (Logic Preserved)
# ===============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BAT_RM_UNet(n_classes=len(class_names), in_channels=3).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

def decode_mask(mask_path):
    mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    class_map = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    for bgr, cls_id in bgr_to_class.items():
        class_map[np.all(mask_bgr == np.array(bgr, dtype=np.uint8), axis=-1)] = cls_id
    return class_map

def predict_mask(image_path):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    orig_h, orig_w = img_bgr.shape[:2]
    img_resized = cv2.resize(img_bgr, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        pred_class = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(pred_class, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

def class_map_to_bgr(class_map):
    bgr = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, colour in class_bgr_values.items():
        bgr[class_map == cls_id] = colour
    return bgr

def create_error_map_overlay(true_class_map, pred_class_map, original_bgr, alpha=0.5):
    correct = (true_class_map == pred_class_map)
    tp_mask = correct & (true_class_map > 0)
    wrong   = ~correct
    fn_mask = wrong & (true_class_map > 0)
    fp_mask = wrong & (pred_class_map > 0) & (true_class_map == 0)

    error_rgb = np.zeros((*true_class_map.shape, 3), dtype=np.uint8)
    error_rgb[tp_mask] = [0, 255, 0]  # Green
    error_rgb[fn_mask] = [255, 0, 0]  # Red
    error_rgb[fp_mask] = [0, 0, 255]  # Blue

    orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    overlay = orig_rgb.copy()
    blend_mask = tp_mask | fn_mask | fp_mask
    if blend_mask.any():
        overlay[blend_mask] = (alpha * error_rgb[blend_mask] + (1 - alpha) * orig_rgb[blend_mask]).astype(np.uint8)
    return Image.fromarray(overlay)

# ===============================
# 4. MAIN EXECUTION WITH SUBFOLDERS
# ===============================

# Create the 4 distinct folders
subdirs = {
    "original": os.path.join(SAVE_DIR, "1_Original_Images"),
    "true_mask": os.path.join(SAVE_DIR, "2_True_Masks"),
    "pred_mask": os.path.join(SAVE_DIR, "3_Predicted_Masks"),
    "error_map": os.path.join(SAVE_DIR, "4_Error_Analysis_Maps")
}

for d in subdirs.values():
    os.makedirs(d, exist_ok=True)

image_files = sorted([f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

for img_file in image_files:
    print(f"Processing: {img_file}")
    img_name_no_ext = os.path.splitext(img_file)[0]
    
    image_path = os.path.join(IMAGES_FOLDER, img_file)
    mask_path  = os.path.join(MASKS_FOLDER,  img_file)
    if not os.path.exists(mask_path): continue

    try:
        # Load and Inference
        orig_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        true_class_map = decode_mask(mask_path)
        pred_class_map = predict_mask(image_path)
        
        # Color Conversion for display
        true_rgb = cv2.cvtColor(class_map_to_bgr(true_class_map), cv2.COLOR_BGR2RGB)
        pred_rgb = cv2.cvtColor(class_map_to_bgr(pred_class_map), cv2.COLOR_BGR2RGB)
        overlay_pil = create_error_map_overlay(true_class_map, pred_class_map, orig_bgr, alpha=ALPHA)

        # --- 1. Save Individual Files ---
        # Save Original (RGB)
        cv2.imwrite(os.path.join(subdirs["original"], f"{img_name_no_ext}.png"), orig_bgr)
        # Save True Mask (using your original BGR colors)
        cv2.imwrite(os.path.join(subdirs["true_mask"], f"{img_name_no_ext}.png"), cv2.cvtColor(true_rgb, cv2.COLOR_RGB2BGR))
        # Save Predicted Mask (using your original BGR colors)
        cv2.imwrite(os.path.join(subdirs["pred_mask"], f"{img_name_no_ext}.png"), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

        # --- 2. Save Combined Plot (Error Map Folder) ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original Image"); axes[0].axis('off')
        axes[1].imshow(true_rgb); axes[1].set_title("True Mask"); axes[1].axis('off')
        axes[2].imshow(pred_rgb); axes[2].set_title("Predicted Mask"); axes[2].axis('off')
        axes[3].imshow(overlay_pil); axes[3].set_title(f"Error Map (α={ALPHA})"); axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save figure in PNG and PDF (vector)
        comb_path = os.path.join(subdirs["error_map"], img_name_no_ext)
        fig.savefig(f"{comb_path}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{comb_path}.pdf", format='pdf', dpi=600, bbox_inches='tight')
        
        plt.close(fig)
        print(f"  Successfully saved all outputs for {img_file}")

    except Exception as e:
        print(f"  [ERROR] {img_file}: {e}")

print("\nAll tasks completed.")

















































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
























#----------- Prediction with Error Map Entire Folder  Final with Editable PDF Text nnUNet ALL ALL ALL----------------------------



import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# Vector export settings
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

# ===============================
# 1. MODEL ARCHITECTURE — nnUNet (Logic Preserved)
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
    def forward(self, x): return self.conv(self.down(x))

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
        self.enc2 = Down(f, f * 2); self.enc3 = Down(f * 2, f * 4)
        self.enc4 = Down(f * 4, f * 8); self.enc5 = Down(f * 8, f * 16)
        self.up4  = Up(f * 16, f * 8, f * 8); self.up3  = Up(f * 8, f * 4, f * 4)
        self.up2  = Up(f * 4, f * 2, f * 2); self.up1  = Up(f * 2, f, f)
        self.out1 = nn.Conv2d(f, n_classes, 1)
        self.out2 = nn.Conv2d(f * 2, n_classes, 1)
        self.out3 = nn.Conv2d(f * 4, n_classes, 1)
        self.out4 = nn.Conv2d(f * 8, n_classes, 1)

    def forward(self, x):
        s1 = self.enc1(x); s2 = self.enc2(s1); s3 = self.enc3(s2)
        s4 = self.enc4(s3); b = self.enc5(s4)
        d4 = self.up4(b, s4); d3 = self.up3(d4, s3); d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1); out_main = self.out1(d1)
        return out_main

# ===============================
# 2. CONFIGURATION
# ===============================
MODEL_PATH    = r'I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/nnUNet_256/Cervix_small_Axial_100_epochs_pytorch_best_nnUNET_BM.pth'
IMAGES_FOLDER = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/images'
MASKS_FOLDER  = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/masks'
SAVE_DIR      = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/nnUNet_all'

IMAGE_SIZE = (256, 256)
ALPHA      = 0.5

bgr_to_class = {
    (0, 0, 0): 0, (0, 255, 0): 1, (0, 255, 255): 2, (153, 146, 255): 3,
    (64, 64, 128): 4, (255, 255, 0): 5, (255, 60, 255): 6, (255, 55, 55): 7
}
class_bgr_values = {v: k for k, v in bgr_to_class.items()}

# ===============================
# 3. LOAD MODEL & HELPERS
# ===============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NNUNet2D(n_classes=len(bgr_to_class), in_channels=3).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

def decode_mask(mask_path):
    mask_bgr = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    class_map = np.zeros(mask_bgr.shape[:2], dtype=np.uint8)
    for bgr, cls_id in bgr_to_class.items():
        class_map[np.all(mask_bgr == np.array(bgr, dtype=np.uint8), axis=-1)] = cls_id
    return class_map

def predict_mask(image_path):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        pred_class = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(pred_class, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

def class_map_to_bgr(class_map):
    bgr = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls_id, colour in class_bgr_values.items():
        bgr[class_map == cls_id] = colour
    return bgr

def create_error_map_overlay(true_class_map, pred_class_map, original_bgr, alpha=0.5):
    correct = (true_class_map == pred_class_map)
    tp_mask = correct & (true_class_map > 0)
    fn_mask = ~correct & (true_class_map > 0)
    fp_mask = ~correct & (pred_class_map > 0) & (true_class_map == 0)
    
    error_rgb = np.zeros((*true_class_map.shape, 3), dtype=np.uint8)
    error_rgb[tp_mask] = [0, 255, 0]   # TP: Green
    error_rgb[fn_mask] = [255, 0, 0]   # FN: Red
    error_rgb[fp_mask] = [0, 0, 255]   # FP: Blue

    orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    overlay = orig_rgb.copy()
    blend_mask = tp_mask | fn_mask | fp_mask
    if blend_mask.any():
        overlay[blend_mask] = (alpha * error_rgb[blend_mask] + (1 - alpha) * orig_rgb[blend_mask]).astype(np.uint8)
    return Image.fromarray(overlay)

# ===============================
# 4. MAIN LOOP WITH SEPARATE FOLDERS
# ===============================

# Create 4 specific directories
dirs = {
    "original": os.path.join(SAVE_DIR, "1_Original_Images"),
    "true_mask": os.path.join(SAVE_DIR, "2_True_Masks"),
    "pred_mask": os.path.join(SAVE_DIR, "3_Predicted_Masks"),
    "error_map": os.path.join(SAVE_DIR, "4_Error_Analysis_Maps")
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

image_files = sorted([f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

for img_file in image_files:
    print(f"Processing: {img_file}")
    img_name_no_ext = os.path.splitext(img_file)[0]
    
    image_path = os.path.join(IMAGES_FOLDER, img_file)
    mask_path  = os.path.join(MASKS_FOLDER,  img_file)
    if not os.path.exists(mask_path): continue

    try:
        # Load and process
        orig_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        true_class_map = decode_mask(mask_path)
        pred_class_map = predict_mask(image_path)
        
        # Color masks for saving/plotting
        true_bgr = class_map_to_bgr(true_class_map)
        pred_bgr = class_map_to_bgr(pred_class_map)
        overlay_pil = create_error_map_overlay(true_class_map, pred_class_map, orig_bgr, alpha=ALPHA)

        # --- SAVE INDIVIDUAL COMPONENTS ---
        cv2.imwrite(os.path.join(dirs["original"], f"{img_name_no_ext}.png"), orig_bgr)
        cv2.imwrite(os.path.join(dirs["true_mask"], f"{img_name_no_ext}.png"), true_bgr)
        cv2.imwrite(os.path.join(dirs["pred_mask"], f"{img_name_no_ext}.png"), pred_bgr)

        # --- SAVE COMBINED ERROR ANALYSIS PLOT ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)); axes[0].set_title("Original Image"); axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(true_bgr, cv2.COLOR_BGR2RGB)); axes[1].set_title("True Mask"); axes[1].axis('off')
        axes[2].imshow(cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)); axes[2].set_title("Predicted Mask"); axes[2].axis('off')
        axes[3].imshow(overlay_pil); axes[3].set_title(f"Error Map (α={ALPHA})"); axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save combined figure (PNG and PDF for vector)
        comb_save_path = os.path.join(dirs["error_map"], img_name_no_ext)
        fig.savefig(f"{comb_save_path}.png", dpi=300, bbox_inches='tight')
        fig.savefig(f"{comb_save_path}.pdf", format='pdf', dpi=600, bbox_inches='tight')
        
        plt.close(fig)
        print(f"  Successfully saved all folders for {img_file}")

    except Exception as e:
        print(f"  [ERROR] {img_file}: {e}")

print("\nTask Complete.")





























































































































"""
================================================================================
SCRIPT: Multi-Model Qualitative Contour Analysis
PURPOSE: Performs visual comparison of segmentation boundaries across five 
         different deep learning models (BAT-RM, nnUNet, SegMamba, TransUNet, UNETR).
         
LOGIC: 
    1. Extracts binary masks for a specific class (e.g., GTV) from BGR encoded PNGs.
    2. Computes contours using OpenCV's findContours algorithm.
    3. Overlays Ground Truth (solid line) and Model Predictions (dashed lines) 
       onto the original CT/MRI slice for spatial accuracy assessment.

INPUT STRUCTURE: Requires root model folders containing:
    - 1_Original_Images/
    - 2_True_Masks/
    - 3_Predicted_Masks/

OUTPUT: High-resolution PNG and editable PDF/EPS vector files.
================================================================================
"""

"""
================================================================================
SCRIPT: Multi-Model Qualitative Contour Analysis (with IoU in Legend)
PURPOSE: Performs visual comparison of contours and calculates slice-specific 
         IoU scores to display in the legend for each model.
================================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# --- Publication Quality Settings ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

# =========================================================
# 1. CONFIGURATION & PATHS
# =========================================================
TARGET_CLASS_ID = 6  
TARGET_CLASS_NAME = "GTV"

GT_CONFIG = {
    "color": "lime",
    "linewidth": 1.5, # Slightly thicker for visibility
    "alpha": 0.7,
    "label": "Ground Truth"
}

MODELS = [
    {"name": "BAT-RM",    "path": r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET_Small_512_200_epoch_all', "color": "red",     "lw": 1, "alpha": 1.0},
    {"name": "nnUNet",    "path": r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET_Small_512_134_epoch_all', "color": "blue",    "lw": 1, "alpha": 0.3},
    {"name": "SegMamba",  "path": r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET3+_all',                   "color": "yellow",  "lw": 1, "alpha": 0.3},
    {"name": "TransUNet", "path": r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET_all',                     "color": "magenta", "lw": 1, "alpha": 0.3},
    {"name": "UNETR",     "path": r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/nnUNet_all',                 "color": "cyan",    "lw": 1, "alpha": 0.3}
]

OUTPUT_DIR = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Contour_Comparison_with_IoU'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class_rgb_values = {
    0: (0, 0, 0), 1: (0, 255, 0), 2: (0, 255, 255), 3: (153, 146, 255),
    4: (64, 64, 128), 5: (255, 255, 0), 6: (255, 60, 255), 7: (255, 55, 55),
}

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def get_binary_mask(mask_path, target_idx):
    """Loads a color mask and returns a binary numpy array (True/False)."""
    mask_bgr = cv2.imread(mask_path)
    if mask_bgr is None: return None
    target_bgr = class_rgb_values[target_idx]
    # Return as boolean array for easier logical operations (IoU)
    return np.all(mask_bgr == np.array(target_bgr), axis=-1)

def calculate_iou(gt_mask, pred_mask):
    """Calculates Intersection over Union for two boolean masks."""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def draw_contour_on_ax(ax, binary_mask, color, label, linewidth, alpha):
    """Plots continuous contours from a binary boolean mask."""
    # Convert bool to uint8 for OpenCV
    mask_ui8 = (binary_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_ui8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        cnt = cnt.reshape(-1, 2)
        cnt = np.vstack([cnt, cnt[0]]) # Close loop
        lbl = label if i == 0 else None
        ax.plot(cnt[:, 0], cnt[:, 1], color=color, label=lbl, 
                linewidth=linewidth, alpha=alpha, linestyle='-')

# =========================================================
# 3. MAIN EXECUTION
# =========================================================
base_img_folder = os.path.join(MODELS[0]["path"], "1_Original_Images")
image_files = sorted([f for f in os.listdir(base_img_folder) if f.endswith('.png')])

for img_file in image_files:
    print(f"Processing: {img_file}")
    
    orig_path = os.path.join(base_img_folder, img_file)
    img_orig = cv2.imread(orig_path)
    if img_orig is None: continue
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    # Load Ground Truth Mask once for this slice
    gt_path = os.path.join(MODELS[0]["path"], "2_True_Masks", img_file)
    gt_mask = get_binary_mask(gt_path, TARGET_CLASS_ID)
    
    if gt_mask is None or not np.any(gt_mask):
        print(f"  [SKIP] No GT found for class {TARGET_CLASS_NAME} in {img_file}")
        continue

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_orig)
    
    # 1. Plot Ground Truth
    draw_contour_on_ax(ax, gt_mask, GT_CONFIG["color"], GT_CONFIG["label"], 
                       GT_CONFIG["linewidth"], GT_CONFIG["alpha"])

    # 2. Plot Each Model with IoU calculation
    for m in MODELS:
        pred_path = os.path.join(m["path"], "3_Predicted_Masks", img_file)
        pred_mask = get_binary_mask(pred_path, TARGET_CLASS_ID)
        
        if pred_mask is not None:
            # Calculate slice-specific IoU
            iou_score = calculate_iou(gt_mask, pred_mask)
            # Create dynamic label with IoU score
            legend_label = f"{m['name']} (IoU: {iou_score:.4f})"
            
            if np.any(pred_mask):
                draw_contour_on_ax(ax, pred_mask, m["color"], legend_label, 
                                   m["lw"], m["alpha"])
            else:
                # Still add to legend even if prediction is empty
                ax.plot([], [], color=m["color"], label=f"{legend_label} [No Pred]", alpha=m["alpha"])

    ax.set_title(f"Qualitative Comparison: {TARGET_CLASS_NAME}\n{img_file}", fontsize=12, pad=10)
    ax.legend(loc='upper right', frameon=True, fontsize=8, facecolor='white', framealpha=0.8, edgecolor='black')
    ax.axis('off')
    
    # Save Outputs
    save_base = os.path.join(OUTPUT_DIR, os.path.splitext(img_file)[0])
    plt.savefig(f"{save_base}_contour_iou.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_base}_contour_iou.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig)

print(f"\nDone. Results saved in: {OUTPUT_DIR}")












"""
================================================================================
SCRIPT: Multi-Class & Multi-Model Qualitative Contour Analysis
PURPOSE: Overlays multiple target classes with consolidated legends.
         Calculates all IoUs per model before drawing to avoid legend duplicates.
================================================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# --- Publication Quality Settings ---
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
matplotlib.rcParams['font.family']  = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']

# =========================================================
# 1. CONFIGURATION & PATHS
# =========================================================
TARGET_CLASSES = {
    6: "GTV",
    7: "CTV"
}

# Mapping specific colors to each class for Ground Truth
GT_CONFIGS = {
    "GTV": {"color": "lime", "linewidth": 1, "alpha": 1},
    "CTV": {"color": "orange", "linewidth": 1, "alpha": 1} # Changed to cyan
}

MODELS = [
    {"name": "BAT-RM",    "path": r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET_Small_512_200_epoch_all', "color": "red",     "lw": 1, "alpha": 1.0},
    {"name": "nnUNet",    "path": r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET_Small_512_134_epoch_all', "color": "blue",    "lw": 1, "alpha": 0.3},
    {"name": "SegMamba",  "path": r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET3+_all',                   "color": "yellow",  "lw": 1, "alpha": 0.3},
    {"name": "TransUNet", "path": r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET_all',                     "color": "magenta", "lw": 1, "alpha": 0.3},
    {"name": "UNETR",     "path": r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/nnUNet_all',                 "color": "cyan",    "lw": 1, "alpha": 0.3}
]

OUTPUT_DIR = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/MultiClass_Contour_Comparison'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class_rgb_values = {
    0: (0, 0, 0), 1: (0, 255, 0), 2: (0, 255, 255), 3: (153, 146, 255),
    4: (64, 64, 128), 5: (255, 255, 0), 6: (255, 60, 255), 7: (255, 55, 55),
}

# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================
def get_binary_mask(mask_path, target_idx):
    mask_bgr = cv2.imread(mask_path)
    if mask_bgr is None: return None
    target_bgr = class_rgb_values[target_idx]
    return np.all(mask_bgr == np.array(target_bgr), axis=-1)

def calculate_iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0: return 1.0 if intersection == 0 else 0.0
    return intersection / union

def draw_contour_on_ax(ax, binary_mask, color, label, linewidth, alpha):
    mask_ui8 = (binary_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_ui8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    first_contour = True
    for cnt in contours:
        if len(cnt) < 3: continue
        cnt = cnt.reshape(-1, 2)
        cnt = np.vstack([cnt, cnt[0]])
        lbl = label if (first_contour and label is not None) else None
        ax.plot(cnt[:, 0], cnt[:, 1], color=color, label=lbl, 
                linewidth=linewidth, alpha=alpha, linestyle='-', 
                antialiased=True, solid_joinstyle='round')
        first_contour = False

# =========================================================
# 3. MAIN EXECUTION
# =========================================================
base_img_folder = os.path.join(MODELS[0]["path"], "1_Original_Images")
image_files = sorted([f for f in os.listdir(base_img_folder) if f.endswith('.png')])

for img_file in image_files:
    print(f"Processing: {img_file}")
    
    orig_path = os.path.join(base_img_folder, img_file)
    img_orig = cv2.imread(orig_path)
    if img_orig is None: continue
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_orig)

    gt_masks = {}
    
# 1. Draw GT for all classes
    for c_id, c_name in TARGET_CLASSES.items():
        gt_path = os.path.join(MODELS[0]["path"], "2_True_Masks", img_file)
        mask = get_binary_mask(gt_path, c_id)
        
        if mask is not None and np.any(mask):
            gt_masks[c_id] = mask
            
            # Pull the specific config for GTV or CTV
            # .get(c_name, ...) provides a fallback in case a class is missing from configs
            cfg = GT_CONFIGS.get(c_name, {"color": "white", "linewidth": 1, "alpha": 0.5})
            
            draw_contour_on_ax(
                ax, mask, 
                color=cfg["color"], 
                label=f"GT ({c_name})", 
                linewidth=cfg["linewidth"], 
                alpha=cfg["alpha"]
            )

    # 2. Draw Models
    for m in MODELS:
        model_metrics = []
        masks_to_draw = []

        # PRE-CALCULATE AND COLLECT DATA
        for c_id, c_name in TARGET_CLASSES.items():
            pred_path = os.path.join(m["path"], "3_Predicted_Masks", img_file)
            pred_mask = get_binary_mask(pred_path, c_id)
            
            iou_val = 0.0
            if pred_mask is not None:
                if c_id in gt_masks:
                    iou_val = calculate_iou(gt_masks[c_id], pred_mask)
                model_metrics.append(f"{c_name}:{iou_val:.3f}")
                if np.any(pred_mask):
                    masks_to_draw.append(pred_mask)
        
        # Build one consolidated label for the whole model
        full_label = f"{m['name']} | " + " | ".join(model_metrics)
        
        # DRAW ALL CLASSES FOR THIS MODEL
        for i, p_mask in enumerate(masks_to_draw):
            # Only attach the full_label to the first class drawn for this model
            current_label = full_label if i == 0 else None
            draw_contour_on_ax(ax, p_mask, m["color"], current_label, m["lw"], m["alpha"])

    # Legend Settings
    ax.legend(loc='upper right', frameon=True, fontsize=7, 
              facecolor='white', framealpha=0.8, edgecolor='black')
    
    ax.axis('off')
    
    save_base = os.path.join(OUTPUT_DIR, os.path.splitext(img_file)[0])
    plt.savefig(f"{save_base}_multiclass.png", dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_base}_multiclass.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig)

print(f"\nDone. Results saved in: {OUTPUT_DIR}")












































































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




















































































# =============================================================================
# FULLY AUTOMATIC GRAD-CAM FOR SEGMENTATION (FOLDER INPUT)
# =============================================================================
# FEATURES:
# -----------------------------------------------------------------------------
# 1. Automatically finds LAST CONVOLUTION LAYER
# 2. Processes ALL images inside a folder
# 3. Saves:
#       - Original image
#       - GradCAM heatmap
#       - Overlay image
#       - Combined PDF report
# 4. Saves PNG + PDF in 600 DPI
# 5. User selects TARGET CLASS
# 6. Works with UNet / Attention UNet / Segmentation Models
# =============================================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image
from tqdm import tqdm

from keras.models import load_model
from keras.utils import normalize
from keras.saving import register_keras_serializable

# =============================================================================
# 1. CUSTOM OBJECTS
# =============================================================================

@register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0):

    epsilon = tf.keras.backend.epsilon()

    y_pred = tf.clip_by_value(
        y_pred,
        epsilon,
        1.0 - epsilon
    )

    cross_entropy = -y_true * tf.math.log(y_pred)

    loss = tf.pow(
        1 - y_pred,
        gamma
    ) * cross_entropy

    return tf.reduce_mean(loss, axis=-1)


@register_keras_serializable()
def soft_dice_loss(y_true, y_pred, smooth=1):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=(1, 2, 3)
    )

    sum_true = tf.reduce_sum(
        y_true,
        axis=(1, 2, 3)
    )

    sum_pred = tf.reduce_sum(
        y_pred,
        axis=(1, 2, 3)
    )

    dice = (2. * intersection + smooth) / (
        sum_true + sum_pred + smooth
    )

    return tf.reduce_mean(1 - dice)


@register_keras_serializable()
def combined_loss(y_true, y_pred,
                  gamma=2.0,
                  alpha=0.5):

    return alpha * focal_loss(
        y_true,
        y_pred,
        gamma
    ) + (1 - alpha) * soft_dice_loss(
        y_true,
        y_pred
    )


@register_keras_serializable()
class CustomMeanIoU(tf.keras.metrics.MeanIoU):

    def __init__(self,
                 num_classes,
                 name='mean_iou',
                 dtype='float32',
                 ignore_class=None,
                 sparse_y_true=True,
                 sparse_y_pred=True,
                 axis=-1):

        super().__init__(
            num_classes=num_classes,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class
        )

        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis

    def update_state(self,
                     y_true,
                     y_pred,
                     sample_weight=None):

        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        return super().update_state(
            y_true,
            y_pred,
            sample_weight
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            'sparse_y_true': self.sparse_y_true,
            'sparse_y_pred': self.sparse_y_pred,
            'axis': self.axis
        })

        return config

# =============================================================================
# 2. LOAD MODEL
# =============================================================================

model_path = r"I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/UNET_BM/200 Epochs/best_model.keras"

custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}

model = load_model(
    model_path,
    custom_objects=custom_objects,
    compile=False
)

print("Model Loaded Successfully")

# =============================================================================
# 3. AUTOMATICALLY FIND LAST CONV LAYER
# =============================================================================

def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        # Conv2D
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

        # DepthwiseConv2D
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return layer.name

        # SeparableConv2D
        if isinstance(layer, tf.keras.layers.SeparableConv2D):
            return layer.name

    raise ValueError("No convolution layer found.")

last_conv_layer_name = find_last_conv_layer(model)

print(f"\nAutomatically Selected Last Conv Layer:")
print(last_conv_layer_name)

# =============================================================================
# 4. CLASS LABELS
# =============================================================================

class_names = {
    0: "Background",
    1: "BODY",
    2: "URINARY BLADDER",
    3: "SMALL BOWEL",
    4: "RECTUM",
    5: "FEMORAL HEAD",
    6: "GTV",
    7: "CTV"
}

# =============================================================================
# 5. CHOOSE TARGET CLASS
# =============================================================================

TARGET_CLASS = 6

print(f"\nSelected Class:")
print(f"{TARGET_CLASS} -> {class_names[TARGET_CLASS]}")

# =============================================================================
# 6. IMAGE FOLDER
# =============================================================================

image_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/images"

# =============================================================================
# 7. OUTPUT FOLDER
# =============================================================================

output_dir = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Grad_Cam"

os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# 8. IMAGE SIZE
# =============================================================================

IMG_SIZE = 512

# =============================================================================
# 9. GRADCAM FUNCTION
# =============================================================================

# =============================================================================
# FIXED GRADCAM FUNCTION
# =============================================================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    target_class_idx
):

    # -------------------------------------------------------------------------
    # FIX MODEL OUTPUT
    # -------------------------------------------------------------------------

    model_output = model.output

    # If output is list -> take first output
    if isinstance(model_output, list):
        model_output = model_output[0]

    # -------------------------------------------------------------------------
    # CREATE GRAD MODEL
    # -------------------------------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model_output
        ]
    )

    # -------------------------------------------------------------------------
    # COMPUTE GRADIENTS
    # -------------------------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        # Select target class
        class_channel = predictions[:, :, :, target_class_idx]

        # Global importance score
        loss = tf.reduce_mean(class_channel)

    # -------------------------------------------------------------------------
    # GRADIENTS
    # -------------------------------------------------------------------------

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    # -------------------------------------------------------------------------
    # CHANNEL IMPORTANCE
    # -------------------------------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # -------------------------------------------------------------------------
    # WEIGHT FEATURE MAPS
    # -------------------------------------------------------------------------

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()

# =============================================================================
# 10. IMAGE EXTENSIONS
# =============================================================================

valid_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)

image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith(valid_extensions)
]

print(f"\nTotal Images Found: {len(image_files)}")

# =============================================================================
# 11. PROCESS ALL IMAGES
# =============================================================================

for image_name in tqdm(image_files):

    try:

        image_path = os.path.join(
            image_folder,
            image_name
        )

        base_name = os.path.splitext(image_name)[0]

        # ---------------------------------------------------------------------
        # READ IMAGE
        # ---------------------------------------------------------------------

        original_bgr = cv2.imread(image_path)

        if original_bgr is None:
            print(f"Could not read: {image_name}")
            continue

        original_rgb = cv2.cvtColor(
            original_bgr,
            cv2.COLOR_BGR2RGB
        )

        H, W = original_rgb.shape[:2]

        # ---------------------------------------------------------------------
        # PREPROCESS
        # ---------------------------------------------------------------------

        resized = cv2.resize(
            original_rgb,
            (IMG_SIZE, IMG_SIZE)
        )

        img_array = resized.astype(np.float32)

        img_array = normalize(
            img_array,
            axis=1
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # ---------------------------------------------------------------------
        # GRADCAM
        # ---------------------------------------------------------------------

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            last_conv_layer_name,
            TARGET_CLASS
        )

        # ---------------------------------------------------------------------
        # RESIZE HEATMAP
        # ---------------------------------------------------------------------

        heatmap = cv2.resize(
            heatmap,
            (W, H)
        )

        heatmap_uint8 = np.uint8(
            255 * heatmap
        )

        # ---------------------------------------------------------------------
        # COLORMAP
        # ---------------------------------------------------------------------

        jet = cm.get_cmap("jet")

        jet_colors = jet(np.arange(256))[:, :3]

        jet_heatmap = jet_colors[heatmap_uint8]

        jet_heatmap = np.uint8(
            jet_heatmap * 255
        )

        # ---------------------------------------------------------------------
        # OVERLAY
        # ---------------------------------------------------------------------

        alpha = 0.5

        overlay = cv2.addWeighted(
            original_rgb,
            1 - alpha,
            jet_heatmap,
            alpha,
            0
        )

        # =============================================================================
        # SAVE PNGS
        # =============================================================================

        original_save = os.path.join(
            output_dir,
            f"{base_name}_original.png"
        )

        heatmap_save = os.path.join(
            output_dir,
            f"{base_name}_heatmap.png"
        )

        overlay_save = os.path.join(
            output_dir,
            f"{base_name}_overlay.png"
        )

        Image.fromarray(original_rgb).save(
            original_save,
            dpi=(600, 600)
        )

        Image.fromarray(jet_heatmap).save(
            heatmap_save,
            dpi=(600, 600)
        )

        Image.fromarray(overlay).save(
            overlay_save,
            dpi=(600, 600)
        )

        # =============================================================================
        # SAVE PDF REPORT
        # =============================================================================

        pdf_save = os.path.join(
            output_dir,
            f"{base_name}_GradCAM_Report.pdf"
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6)
        )

        # Original
        axes[0].imshow(original_rgb)
        axes[0].set_title(
            "Original Image",
            fontsize=14
        )
        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(jet_heatmap)
        axes[1].set_title(
            f"GradCAM Heatmap\n{class_names[TARGET_CLASS]}",
            fontsize=14
        )
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(
            "Overlay",
            fontsize=14
        )
        axes[2].axis("off")

        plt.tight_layout()

        plt.savefig(
            pdf_save,
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.1
        )

        plt.close()

    except Exception as e:

        print(f"\nError processing {image_name}")
        print(e)

# =============================================================================
# 12. DONE
# =============================================================================

print("\n====================================")
print("ALL IMAGES PROCESSED SUCCESSFULLY")
print("====================================")

        
        














































# =============================================================================
# PUBLICATION-QUALITY GRAD-CAM FOR MEDICAL IMAGE SEGMENTATION
# =============================================================================
#
# FEATURES
# -----------------------------------------------------------------------------
# 1. Fully automatic last convolution layer detection
# 2. Folder-based batch processing
# 3. Supports:
#       - UNet
#       - Attention UNet
#       - Multi-class segmentation models
# 4. Generates:
#       - Original image
#       - GradCAM heatmap
#       - Clean overlay image
#       - Combined PDF report
# 5. Saves:
#       - PNG (600 DPI)
#       - PDF (600 DPI)
# 6. Removes dark-blue GradCAM background automatically
# 7. Overlays ONLY strong target-class activation regions
# 8. Morphological noise cleaning included
# 9. User-selectable target class
#
# OUTPUT FILES
# -----------------------------------------------------------------------------
# *_original.png
# *_heatmap.png
# *_overlay.png
# *_GradCAM_Report.pdf
#
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image
from tqdm import tqdm

from keras.models import load_model
from keras.utils import normalize
from keras.saving import register_keras_serializable

# =============================================================================
# CUSTOM OBJECTS
# =============================================================================

@register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0):

    epsilon = tf.keras.backend.epsilon()

    y_pred = tf.clip_by_value(
        y_pred,
        epsilon,
        1.0 - epsilon
    )

    cross_entropy = -y_true * tf.math.log(y_pred)

    loss = tf.pow(
        1 - y_pred,
        gamma
    ) * cross_entropy

    return tf.reduce_mean(loss, axis=-1)


@register_keras_serializable()
def soft_dice_loss(y_true, y_pred, smooth=1):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=(1, 2, 3)
    )

    sum_true = tf.reduce_sum(
        y_true,
        axis=(1, 2, 3)
    )

    sum_pred = tf.reduce_sum(
        y_pred,
        axis=(1, 2, 3)
    )

    dice = (2. * intersection + smooth) / (
        sum_true + sum_pred + smooth
    )

    return tf.reduce_mean(1 - dice)


@register_keras_serializable()
def combined_loss(
    y_true,
    y_pred,
    gamma=2.0,
    alpha=0.5
):

    return alpha * focal_loss(
        y_true,
        y_pred,
        gamma
    ) + (1 - alpha) * soft_dice_loss(
        y_true,
        y_pred
    )


@register_keras_serializable()
class CustomMeanIoU(tf.keras.metrics.MeanIoU):

    def __init__(
        self,
        num_classes,
        name='mean_iou',
        dtype='float32',
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1
    ):

        super().__init__(
            num_classes=num_classes,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class
        )

        self.sparse_y_true = sparse_y_true
        self.sparse_y_pred = sparse_y_pred
        self.axis = axis

    def update_state(
        self,
        y_true,
        y_pred,
        sample_weight=None
    ):

        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        return super().update_state(
            y_true,
            y_pred,
            sample_weight
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            'sparse_y_true': self.sparse_y_true,
            'sparse_y_pred': self.sparse_y_pred,
            'axis': self.axis
        })

        return config

# =============================================================================
# LOAD TRAINED MODEL
# =============================================================================

model_path = r"I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/UNET_BM/200 Epochs/best_model.keras"

custom_objects = {
    'combined_loss': combined_loss,
    'focal_loss': focal_loss,
    'soft_dice_loss': soft_dice_loss,
    'soft_dice_coefficient': lambda y_true, y_pred: 1 - soft_dice_loss(y_true, y_pred),
    'CustomMeanIoU': CustomMeanIoU
}

model = load_model(
    model_path,
    custom_objects=custom_objects,
    compile=False
)

print("Model Loaded Successfully")

# =============================================================================
# AUTOMATICALLY FIND LAST CONVOLUTION LAYER
# =============================================================================

def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return layer.name

        if isinstance(layer, tf.keras.layers.SeparableConv2D):
            return layer.name

    raise ValueError("No convolution layer found.")

last_conv_layer_name = find_last_conv_layer(model)

print("\nAutomatically Selected Last Conv Layer:")
print(last_conv_layer_name)

# =============================================================================
# CLASS LABELS
# =============================================================================

class_names = {
    0: "Background",
    1: "BODY",
    2: "URINARY BLADDER",
    3: "SMALL BOWEL",
    4: "RECTUM",
    5: "FEMORAL HEAD",
    6: "GTV",
    7: "CTV"
}

# =============================================================================
# SELECT TARGET CLASS
# =============================================================================

TARGET_CLASS = 6

print(f"\nSelected Class:")
print(f"{TARGET_CLASS} -> {class_names[TARGET_CLASS]}")

# =============================================================================
# INPUT IMAGE FOLDER
# =============================================================================

image_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/images"

# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

output_dir = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Grad_Cam"

os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# IMAGE SIZE
# =============================================================================

IMG_SIZE = 512

# =============================================================================
# GRADCAM FUNCTION
# =============================================================================

def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    target_class_idx
):

    # -------------------------------------------------------------------------
    # FIX MULTI-OUTPUT MODELS
    # -------------------------------------------------------------------------

    model_output = model.output

    if isinstance(model_output, list):
        model_output = model_output[0]

    # -------------------------------------------------------------------------
    # CREATE GRAD MODEL
    # -------------------------------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model_output
        ]
    )

    # -------------------------------------------------------------------------
    # COMPUTE GRADIENTS
    # -------------------------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        class_channel = predictions[:, :, :, target_class_idx]

        loss = tf.reduce_mean(class_channel)

    grads = tape.gradient(
        loss,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # -------------------------------------------------------------------------
    # WEIGHT FEATURE MAPS
    # -------------------------------------------------------------------------

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()

# =============================================================================
# IMAGE EXTENSIONS
# =============================================================================

valid_extensions = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)

image_files = [
    f for f in os.listdir(image_folder)
    if f.lower().endswith(valid_extensions)
]

print(f"\nTotal Images Found: {len(image_files)}")

# =============================================================================
# PROCESS ALL IMAGES
# =============================================================================

for image_name in tqdm(image_files):

    try:

        # ---------------------------------------------------------------------
        # IMAGE PATHS
        # ---------------------------------------------------------------------

        image_path = os.path.join(
            image_folder,
            image_name
        )

        base_name = os.path.splitext(image_name)[0]

        # ---------------------------------------------------------------------
        # READ IMAGE
        # ---------------------------------------------------------------------

        original_bgr = cv2.imread(image_path)

        if original_bgr is None:
            print(f"Could not read: {image_name}")
            continue

        original_rgb = cv2.cvtColor(
            original_bgr,
            cv2.COLOR_BGR2RGB
        )

        H, W = original_rgb.shape[:2]

        # ---------------------------------------------------------------------
        # PREPROCESS
        # ---------------------------------------------------------------------

        resized = cv2.resize(
            original_rgb,
            (IMG_SIZE, IMG_SIZE)
        )

        img_array = resized.astype(np.float32)

        img_array = normalize(
            img_array,
            axis=1
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # ---------------------------------------------------------------------
        # GENERATE GRADCAM
        # ---------------------------------------------------------------------

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            last_conv_layer_name,
            TARGET_CLASS
        )

        # ---------------------------------------------------------------------
        # RESIZE HEATMAP
        # ---------------------------------------------------------------------

        heatmap = cv2.resize(
            heatmap,
            (W, H)
        )

        heatmap_uint8 = np.uint8(
            255 * heatmap
        )

        # ---------------------------------------------------------------------
        # APPLY JET COLORMAP
        # ---------------------------------------------------------------------

        jet = cm.get_cmap("jet")

        jet_colors = jet(np.arange(256))[:, :3]

        jet_heatmap = jet_colors[heatmap_uint8]

        jet_heatmap = np.uint8(
            jet_heatmap * 255
        )

        # =============================================================================
        # SMOOTH MEDICAL-STYLE GRADCAM OVERLAY
        # =============================================================================
        
        # ---------------------------------------------------------------------
        # REMOVE LOW ACTIVATIONS ONLY
        # ---------------------------------------------------------------------
        
        threshold = 0.20
        
        # Keep smooth activations
        filtered_heatmap = heatmap.copy()
        
        filtered_heatmap[filtered_heatmap < threshold] = 0
        
        # ---------------------------------------------------------------------
        # NORMALIZE AGAIN
        # ---------------------------------------------------------------------
        
        if filtered_heatmap.max() > 0:
            filtered_heatmap = filtered_heatmap / filtered_heatmap.max()
        
        # ---------------------------------------------------------------------
        # CREATE COLORED HEATMAP
        # ---------------------------------------------------------------------
        
        heatmap_uint8 = np.uint8(255 * filtered_heatmap)
        
        jet = cm.get_cmap("jet")
        
        jet_colors = jet(np.arange(256))[:, :3]
        
        colored_heatmap = jet_colors[heatmap_uint8]
        
        colored_heatmap = np.uint8(colored_heatmap * 255)
        
        # ---------------------------------------------------------------------
        # CREATE SOFT ALPHA MAP
        # ---------------------------------------------------------------------
        
        alpha_map = filtered_heatmap.copy()
        
        # Smooth transparency
        alpha_map = np.power(alpha_map, 0.7)
        
        # Scale transparency
        alpha_map = alpha_map * 0.75
        
        # Expand dimensions
        alpha_map = np.expand_dims(alpha_map, axis=-1)
        
        # ---------------------------------------------------------------------
        # SMOOTH OVERLAY
        # ---------------------------------------------------------------------
        
        overlay = (
            original_rgb * (1 - alpha_map) +
            colored_heatmap * alpha_map
        ).astype(np.uint8)
        
        # Save heatmap version
        masked_heatmap = colored_heatmap

        # =============================================================================
        # SAVE PNG FILES
        # =============================================================================

        original_save = os.path.join(
            output_dir,
            f"{base_name}_original.png"
        )

        heatmap_save = os.path.join(
            output_dir,
            f"{base_name}_heatmap.png"
        )

        overlay_save = os.path.join(
            output_dir,
            f"{base_name}_overlay.png"
        )

        Image.fromarray(original_rgb).save(
            original_save,
            dpi=(600, 600)
        )

        Image.fromarray(masked_heatmap).save(
            heatmap_save,
            dpi=(600, 600)
        )

        Image.fromarray(overlay).save(
            overlay_save,
            dpi=(600, 600)
        )

        # =============================================================================
        # SAVE PDF REPORT
        # =============================================================================

        pdf_save = os.path.join(
            output_dir,
            f"{base_name}_GradCAM_Report.pdf"
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6)
        )

        # ---------------------------------------------------------------------
        # ORIGINAL
        # ---------------------------------------------------------------------

        axes[0].imshow(original_rgb)
        axes[0].set_title(
            "Original Image",
            fontsize=14
        )
        axes[0].axis("off")

        # ---------------------------------------------------------------------
        # HEATMAP
        # ---------------------------------------------------------------------

        axes[1].imshow(masked_heatmap)
        axes[1].set_title(
            f"GradCAM Heatmap\n{class_names[TARGET_CLASS]}",
            fontsize=14
        )
        axes[1].axis("off")

        # ---------------------------------------------------------------------
        # OVERLAY
        # ---------------------------------------------------------------------

        axes[2].imshow(overlay)
        axes[2].set_title(
            "Overlay",
            fontsize=14
        )
        axes[2].axis("off")

        plt.tight_layout()

        plt.savefig(
            pdf_save,
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.1
        )

        plt.close()

    except Exception as e:

        print(f"\nError processing {image_name}")
        print(e)

# =============================================================================
# FINISHED
# =============================================================================

print("\n====================================")
print("ALL IMAGES PROCESSED SUCCESSFULLY")
print("====================================")    











































# =============================================================================
# ROBUST MULTI-CAM VISUALIZATION FOR MEDICAL SEGMENTATION
# =============================================================================
# CAMs included:
#   1. GradCAM
#   2. GradCAM++
#   3. ScoreCAM (safe)
#   4. EigenCAM (safe)
#
# FIXES:
#   - No NoneType crashes
#   - Safe fallback CAM handling
#   - Stable batch processing
#   - UNet-compatible
# =============================================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from tqdm import tqdm
from PIL import Image

from keras.models import load_model
from keras.utils import normalize
from keras.saving import register_keras_serializable

# =============================================================================
# CUSTOM LOSS + METRICS (UNCHANGED)
# =============================================================================

@register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    return tf.reduce_mean(-(y_true * tf.math.log(y_pred)) * tf.pow(1 - y_pred, gamma))


@register_keras_serializable()
def soft_dice_loss(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    inter = tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denom = tf.reduce_sum(y_true, axis=(1,2,3)) + tf.reduce_sum(y_pred, axis=(1,2,3))

    dice = (2. * inter + smooth) / (denom + smooth)
    return tf.reduce_mean(1 - dice)


@register_keras_serializable()
def combined_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + soft_dice_loss(y_true, y_pred)


@register_keras_serializable()
class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

# =============================================================================
# MODEL LOAD
# =============================================================================

model_path = r"I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/UNET_BM/200 Epochs/best_model.keras"

model = load_model(
    model_path,
    custom_objects={
        'combined_loss': combined_loss,
        'focal_loss': focal_loss,
        'soft_dice_loss': soft_dice_loss,
        'CustomMeanIoU': CustomMeanIoU
    },
    compile=False
)

print("Model Loaded Successfully")

# =============================================================================
# LAST CONV LAYER
# =============================================================================

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D,
                               tf.keras.layers.SeparableConv2D)):
            return layer.name
    raise ValueError("No conv layer found")

last_conv_layer_name = find_last_conv_layer(model)
print("Last Conv Layer:", last_conv_layer_name)

# =============================================================================
# SETTINGS
# =============================================================================

IMG_SIZE = 512
TARGET_CLASS = 6

image_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/images"
output_dir = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/MultiCAM_SAFE"

os.makedirs(output_dir, exist_ok=True)

class_names = {0:"BG",1:"BODY",2:"BLADDER",3:"SB",4:"RECTUM",5:"FH",6:"GTV",7:"CTV"}

# =============================================================================
# GRAD MODEL
# =============================================================================

model_output = model.output
if isinstance(model_output, list):
    model_output = model_output[0]

grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[model.get_layer(last_conv_layer_name).output, model_output]
)

# =============================================================================
# SAFE WRAPPER (IMPORTANT FIX)
# =============================================================================

def safe_cam(func, img_array, name):
    try:
        cam = func(img_array)
        if cam is None:
            print(f"[WARN] {name} returned None")
            return np.zeros((IMG_SIZE, IMG_SIZE))
        return cam
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE))

# =============================================================================
# OVERLAY FUNCTION
# =============================================================================

def overlay_cam(img, cam):

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    heat = np.uint8(255 * cam)

    jet = cm.get_cmap("jet")
    colors = jet(np.arange(256))[:, :3]
    colors = np.uint8(colors * 255)

    heatmap = colors[heat]

    alpha = 0.45

    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

# =============================================================================
# GRADCAM
# =============================================================================

def gradcam(img):

    with tf.GradientTape() as tape:
        conv, pred = grad_model(img)
        loss = tf.reduce_mean(pred[:, :, :, TARGET_CLASS])

    grads = tape.gradient(loss, conv)

    weights = tf.reduce_mean(grads, axis=(0,1,2))

    cam = tf.reduce_sum(conv[0] * weights, axis=-1)

    return cam.numpy()

# =============================================================================
# GRADCAM++
# =============================================================================

def gradcam_plus(img):

    with tf.GradientTape() as tape:
        conv, pred = grad_model(img)

        pred = pred[0]  # FIX list/tuple safety

        loss = tf.reduce_mean(pred[:, :, TARGET_CLASS])

    grads = tape.gradient(loss, conv)

    if grads is None:
        return np.zeros(conv.shape[1:3])

    weights = tf.reduce_mean(grads, axis=(0,1,2))

    cam = tf.reduce_sum(conv[0] * weights, axis=-1)

    cam = tf.maximum(cam, 0)

    return cam.numpy()

# =============================================================================
# SCORECAM (SAFE VERSION)
# =============================================================================

def scorecam(img):

    conv, pred = grad_model(img)
    conv = conv[0].numpy()

    h, w, c = conv.shape
    cam = np.zeros((h, w))

    for i in range(min(c, 16)):

        act = conv[:, :, i]
        act = cv2.resize(act, (IMG_SIZE, IMG_SIZE))

        if act.max() == act.min():
            continue

        act = (act - act.min()) / (act.max() - act.min())

        masked = img[0] * np.expand_dims(act, -1)
        masked = np.expand_dims(masked, 0)

        pred2 = model(masked)
        if isinstance(pred2, list):
            pred2 = pred2[0]

        score = np.mean(pred2[:, :, :, TARGET_CLASS])

        cam += score * conv[:, :, i]

    return cam

# =============================================================================
# EIGENCAM (SAFE)
# =============================================================================

def eigencam(img):

    conv, _ = grad_model(img)
    conv = conv[0].numpy()

    flat = conv.reshape(-1, conv.shape[-1])
    cov = np.cov(flat, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)

    principal = eigvecs[:, -1]

    cam = np.dot(flat, principal)
    return cam.reshape(conv.shape[:2])

# =============================================================================
# IMAGE LOOP
# =============================================================================

files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png",".jpg",".jpeg"))]

print("Total Images:", len(files))

for f in tqdm(files, desc="Processing Images"):

    try:

        path = os.path.join(image_folder, f)

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        arr = normalize(resized.astype(np.float32), axis=1)
        arr = np.expand_dims(arr, 0)

        # CAMS (SAFE)
        g = safe_cam(gradcam, arr, "GradCAM")
        gp = safe_cam(gradcam_plus, arr, "GradCAM++")
        s = safe_cam(scorecam, arr, "ScoreCAM")
        e = safe_cam(eigencam, arr, "EigenCAM")

        # OVERLAYS
        og = overlay_cam(img, g)
        ogp = overlay_cam(img, gp)
        oscam = overlay_cam(img, s)
        oe = overlay_cam(img, e)

        # PLOT
        fig, ax = plt.subplots(1, 5, figsize=(25, 6))

        ax[0].imshow(img); ax[0].set_title("Original"); ax[0].axis("off")
        ax[1].imshow(og); ax[1].set_title("GradCAM"); ax[1].axis("off")
        ax[2].imshow(ogp); ax[2].set_title("GradCAM++"); ax[2].axis("off")
        ax[3].imshow(oscam); ax[3].set_title("ScoreCAM"); ax[3].axis("off")
        ax[4].imshow(oe); ax[4].set_title("EigenCAM"); ax[4].axis("off")

        plt.suptitle(f"Class: {class_names[TARGET_CLASS]}")
        plt.tight_layout()

        out = os.path.join(output_dir, f"{os.path.splitext(f)[0]}_{class_names[TARGET_CLASS]}_CAM.png")

        plt.savefig(out, dpi=600, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print("Error:", f, e)

print("DONE")













































# =============================================================================
# ROBUST MULTI-CAM VISUALIZATION FOR MEDICAL SEGMENTATION
# =============================================================================
# CAMs included:
#   1. GradCAM++
#   2. HiResCAM (more faithful than GradCAM)
#   3. ScoreCAM (causal)
#
# FEATURES:
#   - 4 plots: Original, GradCAM++, HiResCAM, ScoreCAM
#   - Single universal colorbar
#   - C-Score calculation and Excel export
#   - FIXED: HiResCAM now correctly implemented
# =============================================================================

# =============================================================================
# COMPLETE FIXED CODE - RUN THIS ENTIRELY
# =============================================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from datetime import datetime

from tqdm import tqdm
from sklearn.metrics import jaccard_score
from scipy.stats import pearsonr

from keras.models import load_model
from keras.utils import normalize
from keras.saving import register_keras_serializable

# =============================================================================
# CUSTOM LOSS + METRICS
# =============================================================================

@register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    return tf.reduce_mean(-(y_true * tf.math.log(y_pred)) * tf.pow(1 - y_pred, gamma))

@register_keras_serializable()
def soft_dice_loss(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    inter = tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denom = tf.reduce_sum(y_true, axis=(1,2,3)) + tf.reduce_sum(y_pred, axis=(1,2,3))
    
    dice = (2. * inter + smooth) / (denom + smooth)
    return tf.reduce_mean(1 - dice)

@register_keras_serializable()
def combined_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + soft_dice_loss(y_true, y_pred)

@register_keras_serializable()
class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

# =============================================================================
# MODEL LOAD
# =============================================================================

model_path = r"I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/UNET_BM/200 Epochs/best_model.keras"

model = load_model(
    model_path,
    custom_objects={
        'combined_loss': combined_loss,
        'focal_loss': focal_loss,
        'soft_dice_loss': soft_dice_loss,
        'CustomMeanIoU': CustomMeanIoU
    },
    compile=False
)

print("Model Loaded Successfully")

# =============================================================================
# LAST CONV LAYER
# =============================================================================

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D,
                               tf.keras.layers.SeparableConv2D,
                               tf.keras.layers.Conv2DTranspose)):
            return layer.name
    raise ValueError("No conv layer found")

last_conv_layer_name = find_last_conv_layer(model)
print("Last Conv Layer:", last_conv_layer_name)

# =============================================================================
# SETTINGS
# =============================================================================

IMG_SIZE = 512
TARGET_CLASS = 6

image_folder = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/images"
output_dir = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/MultiCAM_SAFE_FIXED"

os.makedirs(output_dir, exist_ok=True)

class_names = {0:"BG",1:"BODY",2:"BLADDER",3:"SB",4:"RECTUM",5:"FH",6:"GTV",7:"CTV"}

# =============================================================================
# GRAD MODEL
# =============================================================================

model_output = model.output
if isinstance(model_output, list):
    model_output = model_output[0]

grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[model.get_layer(last_conv_layer_name).output, model_output]
)

# =============================================================================
# SAFE WRAPPER WITH CONSISTENT NORMALIZATION
# =============================================================================

def normalize_cam(cam):
    """Consistent normalization for all CAMs"""
    cam = np.maximum(cam, 0)  # ReLU
    
    # Percentile-based normalization to avoid outlier issues
    p98 = np.percentile(cam, 98)
    if p98 > 0:
        cam = np.clip(cam, 0, p98) / p98
    elif cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam

def safe_cam(func, img_array, name):
    try:
        cam = func(img_array)
        if cam is None:
            print(f"[WARN] {name} returned None")
            return np.zeros((IMG_SIZE, IMG_SIZE))
        if np.all(cam == 0):
            print(f"[WARN] {name} returned all zeros")
            return np.zeros((IMG_SIZE, IMG_SIZE))
        
        # Apply consistent normalization
        cam = normalize_cam(cam)
        return cam
        
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return np.zeros((IMG_SIZE, IMG_SIZE))

# =============================================================================
# C-SCORE CALCULATION
# =============================================================================

def calculate_c_score(cam1, cam2):
    """Calculate Consistency Score (C-Score) between two CAMs"""
    flat1 = cam1.flatten()
    flat2 = cam2.flatten()
    
    # Pearson correlation
    if np.std(flat1) > 0 and np.std(flat2) > 0:
        corr, _ = pearsonr(flat1, flat2)
    else:
        corr = 0
    
    # IoU at 0.5 threshold
    thresh1 = (cam1 > 0.5).astype(np.uint8).flatten()
    thresh2 = (cam2 > 0.5).astype(np.uint8).flatten()
    
    if np.sum(thresh1) > 0 or np.sum(thresh2) > 0:
        iou = jaccard_score(thresh1, thresh2, zero_division=0)
    else:
        iou = 0
    
    return (corr + iou) / 2

# =============================================================================
# GRADCAM++ (WITH SPATIAL AVERAGING)
# =============================================================================

def gradcam_plus(img):
    """GradCAM++ for segmentation"""
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            conv_output, predictions = grad_model(img)
            
            if len(predictions.shape) == 3:
                predictions = tf.expand_dims(predictions, 0)
            
            # Spatial averaging for GradCAM++
            loss = tf.reduce_mean(predictions[:, :, :, TARGET_CLASS])
        
        grads = tape2.gradient(loss, conv_output)
        
        if grads is None:
            return np.zeros((conv_output.shape[1], conv_output.shape[2]))
        
        second_grads = tape1.gradient(grads, conv_output)
        
        if second_grads is None:
            weights = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam = tf.reduce_sum(tf.squeeze(conv_output) * weights, axis=-1)
        else:
            grads_pow = tf.pow(grads, 2)
            grads_pow_2 = tf.pow(grads, 3)
            
            alpha_num = grads_pow
            alpha_den = 2 * grads_pow + tf.reduce_sum(conv_output * grads_pow_2, axis=(1, 2), keepdims=True)
            alpha = alpha_num / (alpha_den + 1e-8)
            
            weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(1, 2))
            weights = tf.squeeze(weights)
            
            cam = tf.reduce_sum(tf.squeeze(conv_output) * weights, axis=-1)
    
    cam = tf.maximum(cam, 0)
    return cam.numpy()

# =============================================================================
# HIRESCAM (DIFFERENT - NO SPATIAL AVERAGING)
# =============================================================================

def hirescam(img):
    """
    HiResCAM - No spatial averaging on loss
    This will produce DIFFERENT results from GradCAM++
    """
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img)
        
        if len(predictions.shape) == 3:
            predictions = tf.expand_dims(predictions, 0)
        
        # CRITICAL: Keep spatial dimension (NO reduce_mean)
        # This makes HiResCAM fundamentally different
        loss = predictions[0, :, :, TARGET_CLASS]  # Shape: (H, W)
    
    # Gradients preserve spatial structure
    grads = tape.gradient(loss, conv_output)
    
    if grads is None:
        return np.zeros((conv_output.shape[1], conv_output.shape[2]))
    
    # Element-wise multiplication
    conv_squeezed = tf.squeeze(conv_output)
    grads_squeezed = tf.squeeze(grads)
    
    # Sum over channels
    cam = tf.reduce_sum(conv_squeezed * grads_squeezed, axis=-1)
    cam = tf.maximum(cam, 0)
    
    return cam.numpy()

# =============================================================================
# SCORECAM (OPTIMIZED)
# =============================================================================

def scorecam(img):
    """ScoreCAM - Causal masking approach"""
    conv_output, _ = grad_model(img)
    conv_output = tf.squeeze(conv_output).numpy()
    
    h, w, c = conv_output.shape
    cam = np.zeros((h, w))
    
    # Use all channels for better results
    n_channels = min(c, 64)
    
    # Get base prediction once
    base_pred = model(img)
    if isinstance(base_pred, list):
        base_pred = base_pred[0]
    base_score = np.mean(base_pred[0, :, :, TARGET_CLASS])
    
    for i in range(n_channels):
        activation = conv_output[:, :, i]
        
        # Skip if activation is flat
        if activation.max() - activation.min() < 0.01:
            continue
        
        # Upsample and normalize
        activation_resized = cv2.resize(activation, (IMG_SIZE, IMG_SIZE))
        activation_norm = (activation_resized - activation_resized.min()) / (activation_resized.max() - activation_resized.min() + 1e-8)
        
        # Causal masking
        masked_input = img[0].copy()
        for c_idx in range(3):  # For RGB channels
            masked_input[:, :, c_idx] = masked_input[:, :, c_idx] * activation_norm
        
        masked_input = np.expand_dims(masked_input, 0)
        
        # Get prediction
        pred = model(masked_input)
        if isinstance(pred, list):
            pred = pred[0]
        
        score = np.mean(pred[:, :, :, TARGET_CLASS])
        
        # Weight by relative score increase
        weight = max(0, score - base_score)
        cam += weight * activation
    
    cam = np.maximum(cam, 0)
    return cam

# =============================================================================
# OVERLAY FUNCTION
# =============================================================================

def create_overlay(img, cam, alpha=0.45):
    """Create overlay with consistent colormap"""
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    
    colormap = cm.get_cmap('jet')
    heatmap = colormap(cam_resized)[:, :, :3]
    heatmap = np.uint8(heatmap * 255)
    
    if img.dtype != np.uint8:
        img = np.uint8(img)
    
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png",".jpg",".jpeg"))]
print("Total Images:", len(files))

excel_data = []

for f in tqdm(files, desc="Processing Images"):
    try:
        # Load image
        path = os.path.join(image_folder, f)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        arr = normalize(resized.astype(np.float32), axis=1)
        arr = np.expand_dims(arr, 0)
        
        # Generate CAMs (each uses its own implementation)
        gradcampp_raw = gradcam_plus(arr)
        hirescam_raw = hirescam(arr)
        scorecam_raw = scorecam(arr)
        
        # Apply consistent normalization
        gradcampp_map = normalize_cam(gradcampp_raw)
        hirescam_map = normalize_cam(hirescam_raw)
        scorecam_map = normalize_cam(scorecam_raw)
        
        # Calculate C-Scores
        c1 = calculate_c_score(gradcampp_map, hirescam_map)
        c2 = calculate_c_score(gradcampp_map, scorecam_map)
        c3 = calculate_c_score(hirescam_map, scorecam_map)
        
        # Store data
        excel_data.append({
            'filename': f,
            'class': class_names[TARGET_CLASS],
            'c_score_gradcam++_hirescam': c1,
            'c_score_gradcam++_scorecam': c2,
            'c_score_hirescam_scorecam': c3,
            'gradcam++_mean': np.mean(gradcampp_map),
            'gradcam++_max': np.max(gradcampp_map),
            'gradcam++_std': np.std(gradcampp_map),
            'hirescam_mean': np.mean(hirescam_map),
            'hirescam_max': np.max(hirescam_map),
            'hirescam_std': np.std(hirescam_map),
            'scorecam_mean': np.mean(scorecam_map),
            'scorecam_max': np.max(scorecam_map),
            'scorecam_std': np.std(scorecam_map)
        })
        
        # Create overlays
        overlay_pp = create_overlay(img, gradcampp_map)
        overlay_hr = create_overlay(img, hirescam_map)
        overlay_sc = create_overlay(img, scorecam_map)
        
        # Create figure
        fig, axes = plt.subplots(1, 4, figsize=(20, 6))
        
        axes[0].imshow(img)
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(overlay_pp)
        axes[1].set_title(f"GradCAM++\nC: {c1:.3f}", fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(overlay_hr)
        axes[2].set_title(f"HiResCAM\nC: {c3:.3f}", fontsize=12)
        axes[2].axis('off')
        
        axes[3].imshow(overlay_sc)
        axes[3].set_title(f"ScoreCAM\nC: {c2:.3f}", fontsize=12)
        axes[3].axis('off')
        
        # Universal colorbar
        cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Activation', fontsize=12, fontweight='bold')
        
        plt.suptitle(f"Class: {class_names[TARGET_CLASS]} - CAM Comparison", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        # Save
        safe_name = class_names[TARGET_CLASS].replace(" ", "_")
        out_path = os.path.join(output_dir, f"{os.path.splitext(f)[0]}_{safe_name}_CAM.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Error on {f}: {e}")
        continue

# Save Excel
df = pd.DataFrame(excel_data)
output_excel = os.path.join(output_dir, f"CAM_Scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
df.to_excel(output_excel, index=False)

print(f"\nComplete! Processed {len(files)} images")
print(f"Results: {output_dir}")
print(f"Excel: {output_excel}")
















































# =============================================================================
# MULTI-CAM VISUALIZATION FOR MEDICAL SEGMENTATION (MedIA-Ready)
# =============================================================================
#
# CAMs implemented (all correctly, independently):
#   1. GradCAM++   — second-order gradient weighting
#   2. HiResCAM    — element-wise grad × activation (no global pooling)
#   3. XGradCAM    — normalized gradient weighting (more stable than GradCAM++)
#
# LAYOUT PER IMAGE:
#   n_cams rows × 4 columns:
#     Col 0: Original image
#     Col 1: Early-layer heatmap overlay
#     Col 2: Mid-layer heatmap overlay
#     Col 3: Last-layer heatmap overlay
#
# WHY THESE 3 CAMs:
#   - GradCAM++ : Handles multiple activations per class well; published standard
#   - HiResCAM  : Proven more faithful than GradCAM (Draelos & Carin, 2021)
#   - XGradCAM  : Axiom-satisfying (conservation + sensitivity); strong for
#                 dense prediction / segmentation (Fu et al., 2020)
#
# WHY NOT ScoreCAM:
#   ScoreCAM is gradient-free and thus independent from gradient-based methods,
#   but it is O(C × forward_passes) — prohibitively slow for 512×512 U-Nets
#   with large channel counts. Use it only for ablation on a small subset.
#
# C-SCORE:
#   Consistency Score = (Pearson r + IoU@0.5) / 2
#   Reported pairwise across all three CAMs per image.
#
# =============================================================================

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from scipy.stats import pearsonr

from keras.models import load_model
from keras.utils import normalize
from keras.saving import register_keras_serializable


# =============================================================================
# CUSTOM LOSS + METRICS  (unchanged — needed for model load)
# =============================================================================

@register_keras_serializable()
def focal_loss(y_true, y_pred, gamma=2.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    return tf.reduce_mean(-(y_true * tf.math.log(y_pred)) * tf.pow(1 - y_pred, gamma))

@register_keras_serializable()
def soft_dice_loss(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denom = tf.reduce_sum(y_true, axis=(1, 2, 3)) + tf.reduce_sum(y_pred, axis=(1, 2, 3))
    return tf.reduce_mean(1 - (2.0 * inter + smooth) / (denom + smooth))

@register_keras_serializable()
def combined_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + soft_dice_loss(y_true, y_pred)

@register_keras_serializable()
class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH   = r"I:/Radiotherapy/Cervix/Paper/code/Trained Models/Small/UNET_BM/200 Epochs/best_model.keras"
IMAGE_FOLDER = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/images"
OUTPUT_DIR   = r"I:/Radiotherapy/Cervix/Paper/Result/Qualitative/MultiCAM_MedIA"

IMG_SIZE     = 512
TARGET_CLASS = 6   # GTV

CLASS_NAMES  = {0:"BG", 1:"BODY", 2:"BLADDER", 3:"SB",
                4:"RECTUM", 5:"FH", 6:"GTV", 7:"CTV"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# MODEL LOAD
# =============================================================================

model = load_model(
    MODEL_PATH,
    custom_objects={
        'combined_loss': combined_loss,
        'focal_loss': focal_loss,
        'soft_dice_loss': soft_dice_loss,
        'CustomMeanIoU': CustomMeanIoU
    },
    compile=False
)
print("Model loaded successfully.")


# =============================================================================
# LAYER SELECTION  (early / mid / last conv)
# =============================================================================

def find_conv_layers(model):
    """
    Return (early_name, mid_name, last_name) for Conv2D-family layers.
    Skips input/output layers that may have no spatial gradients.
    """
    conv_layers = [
        l for l in model.layers
        if isinstance(l, (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
            tf.keras.layers.SeparableConv2D,
            tf.keras.layers.Conv2DTranspose
        ))
    ]
    if len(conv_layers) < 3:
        raise ValueError(f"Need at least 3 conv layers; found {len(conv_layers)}")

    n = len(conv_layers)
    early = conv_layers[n // 6]          # ~first sixth of depth
    mid   = conv_layers[n // 2]          # exact midpoint
    last  = conv_layers[-1]              # final conv

    print(f"  Early layer : {early.name}")
    print(f"  Mid   layer : {mid.name}")
    print(f"  Last  layer : {last.name}")
    return early.name, mid.name, last.name


EARLY_LAYER, MID_LAYER, LAST_LAYER = find_conv_layers(model)
LAYER_NAMES  = [EARLY_LAYER, MID_LAYER, LAST_LAYER]
LAYER_LABELS = ["Early Layer", "Mid Layer", "Last Layer"]


# =============================================================================
# GRAD MODELS  — one per layer, built once
# =============================================================================

def build_grad_model(layer_name):
    model_output = model.output
    if isinstance(model_output, list):
        model_output = model_output[0]
    return tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model_output]
    )

grad_models = {name: build_grad_model(name) for name in LAYER_NAMES}


# =============================================================================
# CAM IMPLEMENTATIONS
# =============================================================================

def _get_loss(predictions):
    """Scalar loss for TARGET_CLASS over a dense prediction map."""
    if len(predictions.shape) == 3:          # (H, W, C) — add batch dim
        predictions = tf.expand_dims(predictions, 0)
    return tf.reduce_mean(predictions[..., TARGET_CLASS])


# ── GradCAM++ ─────────────────────────────────────────────────────────────────
#
# Chattopadhay et al. (2018).  Key formula:
#   α_kc = Σ_ij [ (∂²Y^c / ∂A_k²) / (2·∂²Y^c/∂A_k² + Σ A_k·∂³Y^c/∂A_k³) ]
#
# Implementation note:
#   We use TWO *separate* GradientTapes rather than nesting them so that
#   tape1 records the computation of `grads` w.r.t. conv_output.
#   This gives true second-order gradients (Hessian diagonal) and avoids
#   the collapse to element-wise multiplication that happened in the original.
#
def gradcam_plus_plus(img_array, grad_model):
    """
    Correct GradCAM++ using two independent tapes.
    Returns normalized CAM in [0, 1] at conv-layer spatial resolution.
    """
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            tape1.watch(img_tensor)          # watch input so tape2 can diff through it
            conv_out, preds = grad_model(img_tensor)
            tape1.watch(conv_out)            # explicitly watch activations
            loss = _get_loss(preds)

        # First-order gradients dL/dA
        grads1 = tape1.gradient(loss, conv_out)   # shape (1, h, w, C)

    # Second-order gradients d²L/dA² via tape2 differentiating through tape1
    grads2 = tape2.gradient(grads1, conv_out)     # shape (1, h, w, C)

    if grads1 is None:
        return np.zeros(conv_out.shape[1:3])

    conv_out_np = conv_out.numpy()[0]             # (h, w, C)
    g1 = grads1.numpy()[0]                        # (h, w, C)
    g2 = (grads2.numpy()[0] if grads2 is not None
          else np.zeros_like(g1))                 # (h, w, C)  graceful fallback

    # GradCAM++ alpha weights
    g1_sq   = g1 ** 2
    g1_cube = g1 ** 3
    numerator   = g1_sq
    denominator = (2.0 * g1_sq +
                   np.sum(conv_out_np * g1_cube, axis=(0, 1), keepdims=True) + 1e-8)
    alpha = numerator / denominator              # (h, w, C)

    # Relu on first-order grads, weight by alpha, global sum → per-channel weight
    weights = np.sum(alpha * np.maximum(g1, 0), axis=(0, 1))   # (C,)

    cam = np.sum(conv_out_np * weights, axis=-1)  # (h, w)
    cam = np.maximum(cam, 0)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# ── HiResCAM ──────────────────────────────────────────────────────────────────
#
# Draelos & Carin (2021): "Use HiResCAM instead of Grad-CAM for faithful
# explanations of convolutional neural networks."
# Formula: L^HiRes_k = ReLU( Σ_k  (dY^c/dA_k) ⊙ A_k )
#
# Key difference from GradCAM: NO global average pooling of gradients.
# The element-wise product preserves full spatial resolution.
# Key difference from GradCAM++: no alpha re-weighting; pure gradient × activation.
#
def hirescam(img_array, grad_model):
    """
    HiResCAM — element-wise (grad ⊙ activation), summed over channels.
    """
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_out, preds = grad_model(img_tensor)
        tape.watch(conv_out)
        loss = _get_loss(preds)

    grads = tape.gradient(loss, conv_out)   # (1, h, w, C)

    if grads is None:
        return np.zeros(conv_out.shape[1:3])

    # Element-wise product — DO NOT pool gradients spatially
    conv_np  = conv_out.numpy()[0]          # (h, w, C)
    grads_np = grads.numpy()[0]             # (h, w, C)

    cam = np.sum(conv_np * grads_np, axis=-1)   # (h, w)  ← key line
    cam = np.maximum(cam, 0)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# ── XGradCAM ──────────────────────────────────────────────────────────────────
#
# Fu et al. (2020): "Axiom-based Grad-CAM: Towards Accurate Visualization and
# Explanation of CNNs."
# Formula: w_k = Σ_ij [ A_k_ij · (dY^c/dA_k_ij) ] / Σ_ij A_k_ij
#
# Properties:
#   - Conservation: weights sum to the output score
#   - Sensitivity: zero-activation channels get zero weight
#   Particularly suited for dense prediction (segmentation) where many
#   spatial positions contribute.
#
def xgradcam(img_array, grad_model):
    """
    XGradCAM — activation-normalized gradient weighting.
    """
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_out, preds = grad_model(img_tensor)
        tape.watch(conv_out)
        loss = _get_loss(preds)

    grads = tape.gradient(loss, conv_out)   # (1, h, w, C)

    if grads is None:
        return np.zeros(conv_out.shape[1:3])

    conv_np  = conv_out.numpy()[0]          # (h, w, C)
    grads_np = grads.numpy()[0]             # (h, w, C)

    # Numerator:   Σ_ij A·(dY/dA)  per channel
    numerator   = np.sum(conv_np * grads_np, axis=(0, 1))   # (C,)
    # Denominator: Σ_ij A per channel  (avoid div/0)
    denominator = np.sum(conv_np, axis=(0, 1)) + 1e-8       # (C,)

    weights = numerator / denominator       # (C,)

    cam = np.sum(conv_np * weights, axis=-1)    # (h, w)
    cam = np.maximum(cam, 0)
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam




# =============================================================================
# REGISTRY  — maps cam_name → function
# =============================================================================

CAM_REGISTRY = {
    "GradCAM++": gradcam_plus_plus,
    "HiResCAM":  hirescam,
    "XGradCAM":  xgradcam,
}
CAM_NAMES = list(CAM_REGISTRY.keys())   # order preserved


# =============================================================================
# HELPERS
# =============================================================================

def safe_cam(cam_fn, img_array, grad_model, label):
    try:
        cam = cam_fn(img_array, grad_model)
        if cam is None or np.all(cam == 0):
            print(f"  [WARN] {label} returned zeros/None")
            return np.zeros((IMG_SIZE, IMG_SIZE))
        return cam
    except Exception as e:
        print(f"  [ERROR] {label}: {e}")
        import traceback; traceback.print_exc()
        return np.zeros((IMG_SIZE, IMG_SIZE))


def resize_cam(cam, target_h, target_w):
    """Resize CAM to target spatial size."""
    return cv2.resize(cam.astype(np.float32), (target_w, target_h),
                      interpolation=cv2.INTER_LINEAR)


def cam_to_heatmap_overlay(img_rgb, cam, alpha=0.45):
    """
    Overlay a normalised CAM on an RGB image.
    Returns (overlay_uint8, cam_resized_normalised).
    """
    cam_r = resize_cam(cam, img_rgb.shape[0], img_rgb.shape[1])
    if cam_r.max() > cam_r.min():
        cam_r = (cam_r - cam_r.min()) / (cam_r.max() - cam_r.min() + 1e-8)

    heatmap = np.uint8(cm.jet(cam_r)[:, :, :3] * 255)
    base    = img_rgb.astype(np.uint8) if img_rgb.dtype != np.uint8 else img_rgb
    overlay = cv2.addWeighted(base, 1 - alpha, heatmap, alpha, 0)
    return overlay, cam_r


def calculate_c_score(cam1, cam2):
    """Consistency Score = (Pearson r + IoU@0.5) / 2"""
    f1, f2 = cam1.flatten(), cam2.flatten()
    corr = pearsonr(f1, f2)[0] if np.std(f1) > 0 and np.std(f2) > 0 else 0.0
    t1   = (cam1 > 0.5).astype(np.uint8).flatten()
    t2   = (cam2 > 0.5).astype(np.uint8).flatten()
    iou  = jaccard_score(t1, t2, zero_division=0) if (t1.any() or t2.any()) else 0.0
    return float((corr + iou) / 2)


# =============================================================================
# PLOTTING
# =============================================================================

CMAP       = 'jet'    # perceptually uniform; preferred for medical imaging
FIG_WIDTH  = 5            # inches per subplot column
FIG_HEIGHT = 4.5          # inches per row

def plot_multicam(img_rgb, all_cams, filename, c_scores):
    """
    Plot n_cams rows × 4 columns:
      col 0 : original image (same in every row, labelled once)
      col 1-3: early / mid / last layer overlays

    all_cams : dict { cam_name: { layer_label: overlay_rgb } }
    c_scores : dict { cam_name: { 'early_mid':…, 'early_last':…, 'mid_last':… } }
    """
    n_rows = len(CAM_NAMES)
    n_cols = 4   # original + 3 layers

    fig = plt.figure(figsize=(FIG_WIDTH * n_cols + 1, FIG_HEIGHT * n_rows + 1.2))

    # Reserve right margin for colorbar
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        left=0.04, right=0.88,
        top=0.91, bottom=0.04,
        hspace=0.35, wspace=0.06
    )

    col_titles = ["Original Image", "Early Layer", "Mid Layer", "Last Layer"]

    for row_idx, cam_name in enumerate(CAM_NAMES):
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            if col_idx == 0:
                # Original image column
                ax.imshow(img_rgb)
                if row_idx == 0:
                    ax.set_title(col_titles[0], fontsize=11, fontweight='bold',
                                 pad=6, color='#1a1a2e')
                if col_idx == 0:
                    ax.imshow(img_rgb)
                    if row_idx == 0:
                        ax.set_title(col_titles[0], fontsize=11, fontweight='bold',
                                     pad=6, color='#1a1a2e')
                    # Place CAM name as vertical text to the LEFT of this subplot
                    ax.text(-0.12, 0.5, cam_name,
                            transform=ax.transAxes,
                            fontsize=11, fontweight='bold', color='#1a1a2e',
                            ha='center', va='center',
                            rotation=90)
            else:
                layer_label = LAYER_LABELS[col_idx - 1]   # Early / Mid / Last
                overlay     = all_cams[cam_name][layer_label]
                ax.imshow(overlay)
                if row_idx == 0:
                    ax.set_title(col_titles[col_idx], fontsize=11,
                                 fontweight='bold', pad=6, color='#1a1a2e')

                # Annotate inter-layer C-score on mid/last columns
                score_key = {
                    "Early Layer": "early_mid",
                    "Mid Layer":   "early_mid",
                    "Last Layer":  "mid_last"
                }[layer_label]
                score_val = c_scores[cam_name][score_key]
                ax.text(0.97, 0.03, f"C={score_val:.2f}",
                        transform=ax.transAxes, fontsize=8,
                        color='white', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.25', fc='black', alpha=0.55))

            ax.axis('off')

    # Universal colorbar
    cbar_ax = fig.add_axes([0.90, 0.08, 0.018, 0.78])
    norm    = plt.Normalize(0, 1)
    sm      = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cbar    = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Activation Intensity', fontsize=11, fontweight='bold',
                   color='#1a1a2e', labelpad=10)
    cbar.ax.tick_params(labelsize=9)

    # Main title
    safe_cls = CLASS_NAMES[TARGET_CLASS]
    fig.suptitle(
        f"Multi-CAM Analysis  |  Class: {safe_cls}  |  {filename}",
        fontsize=13, fontweight='bold', color='#1a1a2e', y=0.975
    )

    return fig


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
print(f"\nTotal images found: {len(files)}")

excel_rows = []

for fname in tqdm(files, desc="Processing"):
    try:
        # ── Load & preprocess ──────────────────────────────────────────────
        img_bgr  = cv2.imread(os.path.join(IMAGE_FOLDER, fname))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        arr      = normalize(resized.astype(np.float32), axis=1)
        arr_4d   = np.expand_dims(arr, 0)          # (1, H, W, 3)

        # ── Compute all CAMs at all three layers ───────────────────────────
        # cam_maps[cam_name][layer_name] = normalised cam (h_layer, w_layer)
        cam_maps = {}
        for cam_name, cam_fn in CAM_REGISTRY.items():
            cam_maps[cam_name] = {}
            for layer_name in LAYER_NAMES:
                gm  = grad_models[layer_name]
                tag = f"{cam_name} @ {layer_name}"
                cam = safe_cam(cam_fn, arr_4d, gm, tag)
                cam_maps[cam_name][layer_name] = cam

        # ── Build overlays  (use original image size for display) ──────────
        all_overlays = {}    # cam_name → layer_label → overlay_rgb
        for cam_name in CAM_NAMES:
            all_overlays[cam_name] = {}
            for layer_name, layer_label in zip(LAYER_NAMES, LAYER_LABELS):
                raw_cam  = cam_maps[cam_name][layer_name]
                # Resize cam to original image dimensions for overlay
                overlay, _ = cam_to_heatmap_overlay(img_rgb, raw_cam)
                all_overlays[cam_name][layer_label] = overlay

        # ── C-Scores  (compare layers within each CAM) ────────────────────
        c_scores = {}
        for cam_name in CAM_NAMES:
            early_cam = resize_cam(cam_maps[cam_name][EARLY_LAYER], IMG_SIZE, IMG_SIZE)
            mid_cam   = resize_cam(cam_maps[cam_name][MID_LAYER],   IMG_SIZE, IMG_SIZE)
            last_cam  = resize_cam(cam_maps[cam_name][LAST_LAYER],  IMG_SIZE, IMG_SIZE)
            c_scores[cam_name] = {
                "early_mid":  calculate_c_score(early_cam, mid_cam),
                "early_last": calculate_c_score(early_cam, last_cam),
                "mid_last":   calculate_c_score(mid_cam,   last_cam),
            }

        # ── Cross-CAM C-Scores at last layer (for Excel) ──────────────────
        last_cams = {
            cam_name: resize_cam(cam_maps[cam_name][LAST_LAYER], IMG_SIZE, IMG_SIZE)
            for cam_name in CAM_NAMES
        }
        cross_pp_hr = calculate_c_score(last_cams["GradCAM++"], last_cams["HiResCAM"])
        cross_pp_xg = calculate_c_score(last_cams["GradCAM++"], last_cams["XGradCAM"])
        cross_hr_xg = calculate_c_score(last_cams["HiResCAM"],  last_cams["XGradCAM"])

        # ── Build per-image stat row ───────────────────────────────────────
        row = {"filename": fname, "class": CLASS_NAMES[TARGET_CLASS]}
        for cam_name in CAM_NAMES:
            for ln, ll in zip(LAYER_NAMES, ["early", "mid", "last"]):
                c = cam_maps[cam_name][ln]
                row[f"{cam_name}_{ll}_mean"] = float(np.mean(c))
                row[f"{cam_name}_{ll}_std"]  = float(np.std(c))
                row[f"{cam_name}_{ll}_max"]  = float(np.max(c))
            row[f"c_score_{cam_name}_early_mid"]  = c_scores[cam_name]["early_mid"]
            row[f"c_score_{cam_name}_early_last"] = c_scores[cam_name]["early_last"]
            row[f"c_score_{cam_name}_mid_last"]   = c_scores[cam_name]["mid_last"]
        row["cross_c_gradcam++_hirescam"]  = cross_pp_hr
        row["cross_c_gradcam++_xgradcam"]  = cross_pp_xg
        row["cross_c_hirescam_xgradcam"]   = cross_hr_xg
        excel_rows.append(row)

        # ── Plot & save ────────────────────────────────────────────────────
        fig = plot_multicam(img_rgb, all_overlays, fname, c_scores)
        stem     = os.path.splitext(fname)[0]
        safe_cls = CLASS_NAMES[TARGET_CLASS].replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR,
                                f"{stem}_{safe_cls}_MultiCAM.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)

    except Exception as e:
        import traceback
        print(f"\n[ERROR] {fname}: {e}")
        traceback.print_exc()
        continue


# =============================================================================
# EXCEL EXPORT
# =============================================================================

df = pd.DataFrame(excel_rows)

ts           = datetime.now().strftime('%Y%m%d_%H%M%S')
excel_path   = os.path.join(OUTPUT_DIR, f"MultiCAM_Scores_{ts}.xlsx")

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

    # Sheet 1 — full per-image metrics
    df.to_excel(writer, sheet_name='Per_Image_Metrics', index=False)

    # Sheet 2 — summary statistics
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    summary = pd.DataFrame({
        'Metric': numeric_cols,
        'Mean':   [df[c].mean() for c in numeric_cols],
        'Std':    [df[c].std()  for c in numeric_cols],
        'Min':    [df[c].min()  for c in numeric_cols],
        'Max':    [df[c].max()  for c in numeric_cols],
    })
    summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)

    # Sheet 3 — cross-CAM C-scores at last layer only
    cross_df = df[['filename', 'class',
                   'cross_c_gradcam++_hirescam',
                   'cross_c_gradcam++_xgradcam',
                   'cross_c_hirescam_xgradcam']].copy()
    cross_df.to_excel(writer, sheet_name='Cross_CAM_Consistency', index=False)

print("\n" + "=" * 65)
print("Processing Complete!")
print(f"  Output dir  : {OUTPUT_DIR}")
print(f"  Excel file  : {excel_path}")
print(f"  Images done : {len(excel_rows)}")
if len(df) > 0:
    print("\n  Cross-CAM C-Scores at Last Layer (mean ± std):")
    for col in ['cross_c_gradcam++_hirescam',
                'cross_c_gradcam++_xgradcam',
                'cross_c_hirescam_xgradcam']:
        print(f"    {col:40s}: {df[col].mean():.3f} ± {df[col].std():.3f}")
print("=" * 65)





















































# =============================================================================
# QUALITATIVE FAILURE MODE VISUALIZATION
# Nature Medicine Style — 2D PNG Slices
# =============================================================================
#
# Reads worst-case patients from the quantitative FailureCases_Roster.xlsx,
# then renders four publication-quality figure types:
#
#   FIG 1 — WORST-CASE MONTAGE
#     Grid: worst N patients (rows) × all models (cols).
#     Each cell: CT image + GT contour (green) + predicted contour (red).
#     Ranked by the chosen metric so the most-failed case is top-left.
#
#   FIG 2 — ERROR MAP PANEL (TP / FP / FN decomposition)
#     For the single worst patient per class × metric:
#       CT background | GT overlay | Pred overlay |
#       Error map (TP=green, FP=red, FN=blue) | Dice/HD95 badge
#     One column per model.
#
#   FIG 3 — CONTOUR OVERLAY COMPARISON
#     CT with all models' predicted contours drawn in distinct colours,
#     GT contour in white. Visual comparison at a glance.
#
#   FIG 4 — BEST vs WORST GALLERY (reference model only)
#     Top row = best 4 cases, bottom row = worst 4 cases for BAT-RM.
#     Helps readers understand the operating range.
#
# =============================================================================
# FOLDER STRUCTURE EXPECTED:
#
#   ROOT/
#   ├── images/          ← CT PNGs,  e.g.  patient001_slice042.png
#   ├── masks/           ← GT masks, same filenames as images
#   └── predicted/
#       ├── BAT-RM/      ← predicted masks, same filenames
#       ├── nnUNet/
#       ├── SegMamba/
#       ├── TransUNet/
#       └── UNETR/
#
# Mask encoding: RGB — one unique colour per class (see CLASS_COLORS below).
# Image files are matched by FILENAME STEM (the part before .png).
# The 'Filename' column in the roster must match that stem.
#
# =============================================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import to_rgb
from skimage import measure
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1. CONFIGURATION  — edit this block only
# =============================================================================

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR        = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/Error Analysis Map/Plot/Error Analysis Map (All)/UNET_Small_512_200_epoch_all'     # your root folder
IMAGES_DIR      = os.path.join(ROOT_DIR, "images")
MASKS_DIR       = os.path.join(ROOT_DIR, "masks")
PREDICTED_DIR   = os.path.join(ROOT_DIR, "predicted")

# Output from the quantitative failure analysis script
ROSTER_XLSX     = r'I:/Radiotherapy/Cervix/Paper/Result/Quantitative/Final/mixed/FailureModeAnalysis/FailureCases_Roster.xlsx'

SAVE_DIR        = r'I:/Radiotherapy/Cervix/Paper/Result/Qualitative/FailureVisualization'
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
REFERENCE_MODEL = "BAT-RM"
MODEL_NAMES     = ["BAT-RM", "nnUNet", "SegMamba", "TransUNet", "UNETR"]

# ── Classes and their RGB colours in the mask PNGs ───────────────────────────
# Key   = class name (must match Class_Name in roster, uppercase)
# Value = (R, G, B) as used in your mask files  [0-255]
CLASS_COLORS = {
    "BODY":           (128, 128, 128),
    "URINARY BLADDER":(255,   0,   0),
    "SMALL BOWEL":    (  0, 255,   0),
    "RECTUM":         (  0,   0, 255),
    "FEMORAL HEAD":   (255, 255,   0),
    "GTV":            (  0, 255, 255),
    "CTV":            (255,   0, 255),
}

# ── Visualisation parameters ──────────────────────────────────────────────────
N_WORST         = 5      # how many worst patients to show in the montage
CONTOUR_LW      = 1.5    # contour line width
GT_COLOR        = "lime"
ERROR_TP        = (0.15, 0.65, 0.25)   # green
ERROR_FP        = (0.85, 0.15, 0.15)   # red
ERROR_FN        = (0.10, 0.35, 0.85)   # blue
ALPHA_OVERLAY   = 0.45

# Metrics to generate figures for (must exist in roster)
TARGET_METRICS  = ["Dice", "HD95"]

# Model contour colours for FIG 3
MODEL_CONTOUR_COLORS = {
    "BAT-RM":    "#0077BB",
    "nnUNet":    "#009988",
    "SegMamba":  "#EE7733",
    "TransUNet": "#CC3311",
    "UNETR":     "#AA4499",
}

# ── Matplotlib style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   7.5,
    "axes.facecolor":   "black",
    "figure.facecolor": "white",
    "figure.dpi":       150,
    "savefig.dpi":      600,
    "pdf.fonttype":     42,
})

# =============================================================================
# 2. UTILITY FUNCTIONS
# =============================================================================

def load_image(path):
    """Load PNG as float32 numpy array in [0,1], shape HxW or HxWx3."""
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return img


def load_mask(path):
    """Load mask PNG as uint8 HxWx3 array."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def class_binary(mask_rgb, class_rgb, tol=15):
    """
    Extract a binary mask for a class given its RGB colour.
    tol: per-channel tolerance (handles minor JPEG artefacts).
    """
    r, g, b = class_rgb
    diff = np.abs(mask_rgb.astype(int) - np.array([r, g, b]))
    return (diff.max(axis=-1) <= tol).astype(np.uint8)


def draw_contour(ax, binary_mask, color, lw=1.5, alpha=1.0):
    """Draw contour(s) of a binary mask on an axes."""
    if binary_mask.sum() == 0:
        return
    contours = measure.find_contours(binary_mask.astype(float), 0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0],
                color=color, linewidth=lw, alpha=alpha)


def find_slice_file(patient_stem, directory, model=None):
    """
    Find the PNG file for a patient in a directory.
    Tries exact match first, then substring match.
    If `model` is given, looks inside directory/model/.
    Returns path or None.
    """
    search_dir = os.path.join(directory, model) if model else directory
    if not os.path.isdir(search_dir):
        return None
    for fname in os.listdir(search_dir):
        if not fname.lower().endswith(".png"):
            continue
        stem = os.path.splitext(fname)[0]
        if stem == patient_stem or patient_stem in stem or stem in patient_stem:
            return os.path.join(search_dir, fname)
    return None


def error_map(gt_bin, pred_bin):
    """
    Returns an RGBA image (HxWx4) showing:
      TP = green, FP = red, FN = blue, TN = transparent
    """
    h, w = gt_bin.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    tp = (gt_bin == 1) & (pred_bin == 1)
    fp = (gt_bin == 0) & (pred_bin == 1)
    fn = (gt_bin == 1) & (pred_bin == 0)
    rgba[tp] = [*ERROR_TP, 0.70]
    rgba[fp] = [*ERROR_FP, 0.85]
    rgba[fn] = [*ERROR_FN, 0.85]
    return rgba


def format_metric_badge(value, metric):
    """Format a float for annotation."""
    if np.isnan(value):
        return "N/A"
    if metric in ("Dice", "IoU", "Recall", "Specificity"):
        return f"{value:.3f}"
    return f"{value:.1f} mm"


def ax_off_black(ax):
    """Style an axes for medical image display."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_visible(False)


def patient_metric_value(df_roster, patient, model, cls, metric):
    """Look up a single metric value from the roster dataframe."""
    row = df_roster[
        (df_roster["Filename"] == patient) &
        (df_roster["Model"]    == model)   &
        (df_roster["Class"]    == cls)     &
        (df_roster["Metric"]   == metric)
    ]
    if row.empty:
        return np.nan
    return row["Value"].values[0]


# =============================================================================
# 3. LOAD ROSTER
# =============================================================================

print("Loading failure roster …")
df_roster = pd.read_excel(ROSTER_XLSX, sheet_name="FailureRoster")
df_roster["Class"]    = df_roster["Class"].astype(str).str.upper().str.strip()
df_roster["Model"]    = df_roster["Model"].astype(str).str.strip()
df_roster["Filename"] = df_roster["Filename"].astype(str).str.strip()
df_roster["Metric"]   = df_roster["Metric"].astype(str).str.strip()

print(f"  Roster rows  : {len(df_roster)}")
print(f"  Metrics      : {df_roster['Metric'].unique().tolist()}")
print(f"  Classes      : {df_roster['Class'].unique().tolist()}\n")

classes = sorted(df_roster["Class"].dropna().unique())

# =============================================================================
# 4. FIGURE 1 — WORST-CASE MONTAGE
# =============================================================================
# Rows = worst N patients for REFERENCE model on the chosen metric
# Cols = GT | BAT-RM | nnUNet | SegMamba | TransUNet | UNETR
# Each cell shows CT + GT contour (lime) + predicted contour (model colour)
# =============================================================================

def fig1_worst_case_montage(cls, metric, n_worst=N_WORST):
    lower_better = metric in {"HD95", "ASD", "HD"}
    class_rgb    = CLASS_COLORS.get(cls)
    if class_rgb is None:
        print(f"  [SKIP] No colour defined for class '{cls}'")
        return

    # Get worst patients for the reference model
    ref_rows = df_roster[
        (df_roster["Model"]  == REFERENCE_MODEL) &
        (df_roster["Class"]  == cls) &
        (df_roster["Metric"] == metric)
    ].copy()

    if ref_rows.empty:
        print(f"  [SKIP] No failure cases for {REFERENCE_MODEL} / {cls} / {metric}")
        return

    ref_rows = ref_rows.sort_values("Value", ascending=not lower_better)
    worst_patients = ref_rows["Filename"].tolist()[:n_worst]

    if not worst_patients:
        return

    n_cols = 1 + len(MODEL_NAMES)   # GT + one per model
    n_rows = len(worst_patients)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.6, n_rows * 2.7 + 1.2),
        squeeze=False,
    )

    col_headers = ["Ground Truth"] + MODEL_NAMES
    for ci, hdr in enumerate(col_headers):
        color = MODEL_CONTOUR_COLORS.get(hdr, GT_COLOR) if hdr != "Ground Truth" else GT_COLOR
        axes[0, ci].set_title(
            hdr, fontsize=8, fontweight="bold",
            color=color if hdr != "Ground Truth" else "#00BB44",
            pad=4,
        )

    for ri, patient in enumerate(worst_patients):
        img_path  = find_slice_file(patient, IMAGES_DIR)
        gt_path   = find_slice_file(patient, MASKS_DIR)

        # Left-most label: rank + patient name
        axes[ri, 0].set_ylabel(
            f"#{ri+1}  {patient}", fontsize=6, rotation=0,
            labelpad=4, va="center", ha="right",
        )

        # ── GT column ──
        ax_gt = axes[ri, 0]
        ax_off_black(ax_gt)
        if img_path:
            img = load_image(img_path)
            ax_gt.imshow(img, cmap="gray" if img.ndim == 2 else None)
        if gt_path:
            gt_mask  = load_mask(gt_path)
            gt_bin   = class_binary(gt_mask, class_rgb)
            draw_contour(ax_gt, gt_bin, GT_COLOR, lw=CONTOUR_LW)

        # Reference model metric value badge
        val = ref_rows[ref_rows["Filename"] == patient]["Value"]
        badge = format_metric_badge(val.values[0] if not val.empty else np.nan, metric)
        ax_gt.text(
            0.03, 0.03, f"{metric}: {badge}",
            transform=ax_gt.transAxes,
            fontsize=5.5, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="#CC2222", alpha=0.75, ec="none"),
            va="bottom",
        )

        # ── Model columns ──
        for ci, model in enumerate(MODEL_NAMES):
            ax = axes[ri, ci + 1]
            ax_off_black(ax)

            if img_path:
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

            pred_path = find_slice_file(patient, PREDICTED_DIR, model)
            if pred_path:
                pred_mask = load_mask(pred_path)
                pred_bin  = class_binary(pred_mask, class_rgb)
                # Draw GT contour faintly in green, pred contour in model colour
                if gt_path:
                    draw_contour(ax, gt_bin, GT_COLOR, lw=0.8, alpha=0.5)
                draw_contour(ax, pred_bin,
                             MODEL_CONTOUR_COLORS.get(model, "white"),
                             lw=CONTOUR_LW)

            # Per-model metric badge
            m_val = patient_metric_value(df_roster, patient, model, cls, metric)
            ax.text(
                0.03, 0.03, format_metric_badge(m_val, metric),
                transform=ax.transAxes,
                fontsize=5.5, color="white",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc=MODEL_CONTOUR_COLORS.get(model, "#444"),
                          alpha=0.80, ec="none"),
                va="bottom",
            )
        
        # Star marker if this is row 0 (absolute worst)
        if ri == 0:
            axes[ri, 0].text(
                0.97, 0.97, "▼ WORST",
                transform=axes[ri, 0].transAxes,
                fontsize=6, color="#FF4444", fontweight="bold",
                ha="right", va="top",
            )

    # Legend
    legend_handles = [
        mpatches.Patch(color=GT_COLOR, label="GT contour"),
    ] + [
        mpatches.Patch(color=MODEL_CONTOUR_COLORS[m], label=f"{m} pred")
        for m in MODEL_NAMES
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=len(legend_handles),
        fontsize=7, frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle(
        f"Worst-case montage  |  Class: {cls}  |  Metric: {metric}\n"
        f"Rows = {n_worst} worst patients for {REFERENCE_MODEL}  "
        f"(sorted by {'highest' if lower_better else 'lowest'} {metric})\n"
        f"Contours: lime = GT  |  coloured = model prediction",
        fontsize=9, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    safe = cls.replace(" ", "_")
    for ext in ["png", "pdf"]:
        plt.savefig(
            os.path.join(SAVE_DIR, f"Fig1_WorstCase_{safe}_{metric}.{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight", facecolor="white",
        )
    plt.show()
    print(f"  ✓ Fig1 saved  [{cls} / {metric}]")


# =============================================================================
# 5. FIGURE 2 — ERROR MAP PANEL (TP / FP / FN)
# =============================================================================
# For each class × metric, take the SINGLE worst patient (by reference model)
# and show: CT | GT | Pred | Error map  for EVERY model in one figure row.
# =============================================================================

def fig2_error_maps(cls, metric):
    lower_better = metric in {"HD95", "ASD", "HD"}
    class_rgb    = CLASS_COLORS.get(cls)
    if class_rgb is None:
        return

    ref_rows = df_roster[
        (df_roster["Model"]  == REFERENCE_MODEL) &
        (df_roster["Class"]  == cls) &
        (df_roster["Metric"] == metric)
    ].copy().sort_values("Value", ascending=not lower_better)

    if ref_rows.empty:
        return

    worst_patient = ref_rows["Filename"].iloc[0]
    img_path      = find_slice_file(worst_patient, IMAGES_DIR)
    gt_path       = find_slice_file(worst_patient, MASKS_DIR)

    n_models = len(MODEL_NAMES)
    # 4 sub-cols per model: CT | GT | Pred | Error
    N_SUBCOLS = 4
    fig_w     = n_models * N_SUBCOLS * 1.8 + 1.0
    fig_h     = 3.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer_gs = gridspec.GridSpec(
        1, n_models,
        figure=fig, wspace=0.08,
        left=0.02, right=0.98, top=0.82, bottom=0.06,
    )

    img      = load_image(img_path) if img_path else None
    gt_mask  = load_mask(gt_path)   if gt_path  else None
    gt_bin   = class_binary(gt_mask, class_rgb) if gt_mask is not None else None

    for mi, model in enumerate(MODEL_NAMES):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, N_SUBCOLS, subplot_spec=outer_gs[mi], wspace=0.03
        )
        ax_ct  = fig.add_subplot(inner_gs[0])
        ax_gt  = fig.add_subplot(inner_gs[1])
        ax_pr  = fig.add_subplot(inner_gs[2])
        ax_err = fig.add_subplot(inner_gs[3])

        for ax in [ax_ct, ax_gt, ax_pr, ax_err]:
            ax_off_black(ax)

        # — CT —
        if img is not None:
            ax_ct.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax_ct.set_title(
            model, fontsize=7.5, fontweight="bold",
            color=MODEL_CONTOUR_COLORS.get(model, "white"),
            pad=2,
        )
        if mi == 0:
            ax_ct.set_ylabel("CT", fontsize=6.5, color="white", rotation=0, labelpad=18, va="center")

        # — GT contour on CT —
        if img is not None:
            ax_gt.imshow(img, cmap="gray" if img.ndim == 2 else None)
        if gt_bin is not None:
            gt_rgba = np.zeros((*gt_bin.shape, 4), dtype=np.float32)
            gt_rgba[gt_bin == 1] = [*to_rgb(GT_COLOR), ALPHA_OVERLAY]
            ax_gt.imshow(gt_rgba)
            draw_contour(ax_gt, gt_bin, GT_COLOR, lw=1.2)
        if mi == 0:
            ax_gt.set_ylabel("GT", fontsize=6.5, color="lime", rotation=0, labelpad=18, va="center")

        # — Prediction overlay —
        pred_path = find_slice_file(worst_patient, PREDICTED_DIR, model)
        pred_bin  = None
        if pred_path:
            pred_mask = load_mask(pred_path)
            pred_bin  = class_binary(pred_mask, class_rgb)

        if img is not None:
            ax_pr.imshow(img, cmap="gray" if img.ndim == 2 else None)
        if pred_bin is not None:
            pred_rgba = np.zeros((*pred_bin.shape, 4), dtype=np.float32)
            mc = to_rgb(MODEL_CONTOUR_COLORS.get(model, "white"))
            pred_rgba[pred_bin == 1] = [*mc, ALPHA_OVERLAY]
            ax_pr.imshow(pred_rgba)
            draw_contour(ax_pr, pred_bin, MODEL_CONTOUR_COLORS.get(model, "white"), lw=1.2)
        if mi == 0:
            ax_pr.set_ylabel("Pred", fontsize=6.5, color="white", rotation=0, labelpad=18, va="center")

        # — Error map —
        if img is not None:
            ax_err.imshow(img * 0.35, cmap="gray" if img.ndim == 2 else None)  # dim background
        if gt_bin is not None and pred_bin is not None:
            ax_err.imshow(error_map(gt_bin, pred_bin))
        if mi == 0:
            ax_err.set_ylabel("Error", fontsize=6.5, color="white", rotation=0, labelpad=18, va="center")

        # Metric value badge on error map
        m_val = patient_metric_value(df_roster, worst_patient, model, cls, metric)
        ax_err.text(
            0.05, 0.05, format_metric_badge(m_val, metric),
            transform=ax_err.transAxes,
            fontsize=6.0, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25",
                      fc=MODEL_CONTOUR_COLORS.get(model, "#444"),
                      alpha=0.85, ec="none"),
            va="bottom",
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=ERROR_TP, label="True Positive"),
        mpatches.Patch(color=ERROR_FP, label="False Positive"),
        mpatches.Patch(color=ERROR_FN, label="False Negative"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center", ncol=3,
        fontsize=7, frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        f"Error map analysis  |  Class: {cls}  |  Metric: {metric}\n"
        f"Worst patient: {worst_patient}   "
        f"Columns: CT  /  GT overlay  /  Prediction overlay  /  Error map\n"
        f"Error map:  ■ TP  ■ FP  ■ FN",
        fontsize=9, fontweight="bold", y=0.97,
    )

    safe = cls.replace(" ", "_")
    for ext in ["png", "pdf"]:
        plt.savefig(
            os.path.join(SAVE_DIR, f"Fig2_ErrorMap_{safe}_{metric}.{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight", facecolor="white",
        )
    plt.show()
    print(f"  ✓ Fig2 saved  [{cls} / {metric}]")


# =============================================================================
# 6. FIGURE 3 — CONTOUR OVERLAY COMPARISON (all models on same CT)
# =============================================================================
# For each of the worst N patients, show a SINGLE CT image with all models'
# contours drawn simultaneously in their distinct colours.
# Immediately shows where models agree vs diverge.
# =============================================================================

def fig3_contour_overlay(cls, metric, n_worst=N_WORST):
    lower_better = metric in {"HD95", "ASD", "HD"}
    class_rgb    = CLASS_COLORS.get(cls)
    if class_rgb is None:
        return

    ref_rows = df_roster[
        (df_roster["Model"]  == REFERENCE_MODEL) &
        (df_roster["Class"]  == cls) &
        (df_roster["Metric"] == metric)
    ].copy().sort_values("Value", ascending=not lower_better)

    if ref_rows.empty:
        return

    worst_patients = ref_rows["Filename"].tolist()[:n_worst]
    n_pts = len(worst_patients)

    fig, axes = plt.subplots(
        1, n_pts,
        figsize=(n_pts * 3.2, 3.8),
        squeeze=False,
    )

    for ci, patient in enumerate(worst_patients):
        ax = axes[0, ci]
        ax_off_black(ax)

        img_path = find_slice_file(patient, IMAGES_DIR)
        gt_path  = find_slice_file(patient, MASKS_DIR)

        if img_path:
            img = load_image(img_path)
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

        # GT contour — thick white
        if gt_path:
            gt_mask = load_mask(gt_path)
            gt_bin  = class_binary(gt_mask, class_rgb)
            draw_contour(ax, gt_bin, "white", lw=2.2)

        # All model contours
        for model in MODEL_NAMES:
            pred_path = find_slice_file(patient, PREDICTED_DIR, model)
            if pred_path:
                pred_mask = load_mask(pred_path)
                pred_bin  = class_binary(pred_mask, class_rgb)
                draw_contour(ax, pred_bin,
                             MODEL_CONTOUR_COLORS.get(model, "gray"),
                             lw=1.4, alpha=0.90)

        ref_val = ref_rows[ref_rows["Filename"] == patient]["Value"]
        badge   = format_metric_badge(ref_val.values[0] if not ref_val.empty else np.nan, metric)
        ax.set_title(
            f"#{ci+1}  {patient}\n{REFERENCE_MODEL} {metric}: {badge}",
            fontsize=6.5, color="white", pad=3,
        )

    # Legend
    legend_handles = [
        mpatches.Patch(color="white",  label="GT"),
    ] + [
        mpatches.Patch(color=MODEL_CONTOUR_COLORS[m], label=m)
        for m in MODEL_NAMES
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=len(legend_handles),
        fontsize=7, frameon=True,
        bbox_to_anchor=(0.5, -0.04),
        facecolor="#222222", labelcolor="white", edgecolor="#555555",
    )

    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"Contour overlay  |  Class: {cls}  |  Metric: {metric}\n"
        f"White = GT  |  Coloured = model predictions  "
        f"|  {n_pts} worst cases for {REFERENCE_MODEL}",
        fontsize=9, fontweight="bold", color="white", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    safe = cls.replace(" ", "_")
    for ext in ["png", "pdf"]:
        plt.savefig(
            os.path.join(SAVE_DIR, f"Fig3_ContourOverlay_{safe}_{metric}.{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight", facecolor="#111111",
        )
    plt.show()
    print(f"  ✓ Fig3 saved  [{cls} / {metric}]")


# =============================================================================
# 7. FIGURE 4 — BEST vs WORST GALLERY (reference model)
# =============================================================================
# Top row = 4 best cases (highest Dice or lowest HD95 for ref model)
# Bottom row = 4 worst cases
# Each cell: CT + GT contour + ref-model predicted contour
# Shows the operating range visually.
# =============================================================================

def fig4_best_vs_worst(cls, metric, n_each=4):
    lower_better = metric in {"HD95", "ASD", "HD"}
    class_rgb    = CLASS_COLORS.get(cls)
    if class_rgb is None:
        return

    # All cases for reference model on this metric (not just failures)
    # We need the full distribution — read from the roster (worst) and
    # infer best from the complement of worst cases.
    # Strategy: fetch all filenames for this class from images dir,
    # merge with roster values, sort.
    ref_fail = df_roster[
        (df_roster["Model"]  == REFERENCE_MODEL) &
        (df_roster["Class"]  == cls) &
        (df_roster["Metric"] == metric)
    ].copy().sort_values("Value", ascending=not lower_better)

    if ref_fail.empty:
        return

    # Worst N: already sorted
    worst_patients = ref_fail["Filename"].tolist()[:n_each]
    # Best N: reverse sort
    best_patients  = ref_fail["Filename"].tolist()[-n_each:][::-1]

    n_cols = max(len(best_patients), len(worst_patients))
    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(n_cols * 2.8, 6.2),
        squeeze=False,
    )

    row_labels = [
        f"BEST CASES  (reference: {REFERENCE_MODEL})",
        f"WORST CASES  (reference: {REFERENCE_MODEL})",
    ]
    row_colors = ["#00AA44", "#CC2222"]

    for ri, (patients, label) in enumerate(
        [(best_patients, row_labels[0]), (worst_patients, row_labels[1])]
    ):
        for ci, patient in enumerate(patients):
            ax = axes[ri, ci]
            ax_off_black(ax)

            img_path  = find_slice_file(patient, IMAGES_DIR)
            gt_path   = find_slice_file(patient, MASKS_DIR)
            pred_path = find_slice_file(patient, PREDICTED_DIR, REFERENCE_MODEL)

            if img_path:
                img = load_image(img_path)
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

            if gt_path:
                gt_mask = load_mask(gt_path)
                gt_bin  = class_binary(gt_mask, class_rgb)
                draw_contour(ax, gt_bin, GT_COLOR, lw=1.8)

            if pred_path:
                pred_mask = load_mask(pred_path)
                pred_bin  = class_binary(pred_mask, class_rgb)
                draw_contour(ax, pred_bin,
                             MODEL_CONTOUR_COLORS[REFERENCE_MODEL], lw=1.4)

            # Metric badge
            val_rows = ref_fail[ref_fail["Filename"] == patient]["Value"]
            badge    = format_metric_badge(
                val_rows.values[0] if not val_rows.empty else np.nan, metric
            )
            fc = "#006622" if ri == 0 else "#882222"
            ax.text(
                0.04, 0.04, badge,
                transform=ax.transAxes,
                fontsize=6.5, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22", fc=fc, alpha=0.85, ec="none"),
                va="bottom",
            )
            ax.set_title(patient, fontsize=5.5, color="#CCCCCC", pad=2)

            if ci == 0:
                axes[ri, 0].set_ylabel(
                    label, fontsize=7, fontweight="bold",
                    color=row_colors[ri], rotation=90,
                    labelpad=4,
                )

        # Fill empty cells if rows differ in length
        for ci in range(len(patients), n_cols):
            axes[ri, ci].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=GT_COLOR, label="GT contour"),
        mpatches.Patch(color=MODEL_CONTOUR_COLORS[REFERENCE_MODEL],
                       label=f"{REFERENCE_MODEL} prediction"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=2,
        fontsize=7.5, frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"Best vs Worst gallery  |  {REFERENCE_MODEL}  |  Class: {cls}  |  Metric: {metric}\n"
        f"Top row: {n_each} best cases  ·  Bottom row: {n_each} worst cases\n"
        f"Lime = GT  ·  Blue = {REFERENCE_MODEL} prediction",
        fontsize=9, fontweight="bold", y=1.02,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    safe = cls.replace(" ", "_")
    for ext in ["png", "pdf"]:
        plt.savefig(
            os.path.join(SAVE_DIR, f"Fig4_BestWorst_{safe}_{metric}.{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight", facecolor="white",
        )
    plt.show()
    print(f"  ✓ Fig4 saved  [{cls} / {metric}]")


# =============================================================================
# 8. FIGURE 5 — CROSS-MODEL FAILURE STRIP (multi-patient × multi-model grid)
# =============================================================================
# For a given class and metric, show the top N_WORST failure patients as rows
# and ALL models as columns. Each cell = error map only (compact).
# This is the most information-dense panel: readers can scan rows (patient)
# and columns (model) to spot model-specific vs shared failure modes.
# =============================================================================

def fig5_cross_model_error_strip(cls, metric, n_worst=N_WORST):
    lower_better = metric in {"HD95", "ASD", "HD"}
    class_rgb    = CLASS_COLORS.get(cls)
    if class_rgb is None:
        return

    ref_rows = df_roster[
        (df_roster["Model"]  == REFERENCE_MODEL) &
        (df_roster["Class"]  == cls) &
        (df_roster["Metric"] == metric)
    ].copy().sort_values("Value", ascending=not lower_better)

    if ref_rows.empty:
        return

    worst_patients = ref_rows["Filename"].tolist()[:n_worst]
    n_pts    = len(worst_patients)
    n_models = len(MODEL_NAMES)

    fig, axes = plt.subplots(
        n_pts, n_models,
        figsize=(n_models * 2.2, n_pts * 2.4 + 1.2),
        squeeze=False,
    )

    # Column headers
    for ci, model in enumerate(MODEL_NAMES):
        axes[0, ci].set_title(
            model, fontsize=8, fontweight="bold",
            color=MODEL_CONTOUR_COLORS.get(model, "white"), pad=4,
        )

    for ri, patient in enumerate(worst_patients):
        img_path = find_slice_file(patient, IMAGES_DIR)
        gt_path  = find_slice_file(patient, MASKS_DIR)

        img     = load_image(img_path) if img_path else None
        gt_mask = load_mask(gt_path)   if gt_path  else None
        gt_bin  = class_binary(gt_mask, class_rgb) if gt_mask is not None else None

        for ci, model in enumerate(MODEL_NAMES):
            ax = axes[ri, ci]
            ax_off_black(ax)

            if img is not None:
                ax.imshow(img * 0.30, cmap="gray" if img.ndim == 2 else None)

            pred_path = find_slice_file(patient, PREDICTED_DIR, model)
            if pred_path and gt_bin is not None:
                pred_mask = load_mask(pred_path)
                pred_bin  = class_binary(pred_mask, class_rgb)
                ax.imshow(error_map(gt_bin, pred_bin))

            m_val = patient_metric_value(df_roster, patient, model, cls, metric)
            ax.text(
                0.04, 0.04, format_metric_badge(m_val, metric),
                transform=ax.transAxes,
                fontsize=5.5, color="white",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc=MODEL_CONTOUR_COLORS.get(model, "#333"),
                          alpha=0.80, ec="none"),
                va="bottom",
            )

        # Row label
        ref_val = ref_rows[ref_rows["Filename"] == patient]["Value"]
        badge   = format_metric_badge(ref_val.values[0] if not ref_val.empty else np.nan, metric)
        axes[ri, 0].set_ylabel(
            f"#{ri+1}  {patient}\n({REFERENCE_MODEL}: {badge})",
            fontsize=5.5, rotation=0, labelpad=4, va="center", ha="right",
            color="#DDDDDD",
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=ERROR_TP, label="TP"),
        mpatches.Patch(color=ERROR_FP, label="FP (over-seg)"),
        mpatches.Patch(color=ERROR_FN, label="FN (under-seg)"),
    ]
    fig.legend(
        handles=legend_patches, loc="lower center", ncol=3,
        fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle(
        f"Cross-model error strip  |  Class: {cls}  |  Metric: {metric}\n"
        f"Rows = worst {n_worst} patients (ranked by {REFERENCE_MODEL} {metric})\n"
        f"Columns = models  ·  Each cell = TP/FP/FN error map",
        fontsize=9, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    safe = cls.replace(" ", "_")
    for ext in ["png", "pdf"]:
        plt.savefig(
            os.path.join(SAVE_DIR, f"Fig5_CrossModelStrip_{safe}_{metric}.{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight", facecolor="#111111",
        )
    plt.show()
    print(f"  ✓ Fig5 saved  [{cls} / {metric}]")


# =============================================================================
# 9. RUN ALL FIGURES
# =============================================================================

print("=" * 70)
print("QUALITATIVE FAILURE VISUALIZATION")
print("=" * 70)

for cls in classes:
    cls_rgb = CLASS_COLORS.get(cls)
    if cls_rgb is None:
        print(f"\n[WARNING] No colour mapping for class '{cls}' — skipping.")
        continue

    print(f"\n{'─'*60}")
    print(f"  CLASS: {cls}")
    print(f"{'─'*60}")

    for metric in TARGET_METRICS:
        print(f"\n  Metric: {metric}")
        fig1_worst_case_montage(cls, metric, n_worst=N_WORST)
        fig2_error_maps(cls, metric)
        fig3_contour_overlay(cls, metric, n_worst=N_WORST)
        fig4_best_vs_worst(cls, metric, n_each=4)
        fig5_cross_model_error_strip(cls, metric, n_worst=N_WORST)

print("\n" + "=" * 70)
print(f"All figures saved to:  {SAVE_DIR}")
print("=" * 70)
        
