from keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
from keras.utils import normalize
import os
import glob
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

SIZE_X = 256
SIZE_Y = 256

class DataGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, image_size, class_rgb_array, n_classes):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.class_rgb_array = class_rgb_array
        self.n_classes = n_classes
        self.index = 0  # Initialize index for iteration

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __iter__(self):
        self.indexes = np.arange(len(self.image_paths))
        np.random.shuffle(self.indexes)  # Shuffle if necessary
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        batch_indexes = self.indexes[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return self.__getitem__(batch_indexes[0] // self.batch_size)

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_mask_paths = self.mask_paths[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        masks = []
        
        for img_path, mask_path in zip(batch_image_paths, batch_mask_paths):
            img = cv2.imread(img_path, 1)
            img = cv2.resize(img, self.image_size)  # Uncomment if resizing is needed
            
            mask = cv2.imread(mask_path, 1)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)  # Uncomment if resizing is needed
            
            # Label encoding the mask
            mask_encoded = self.label_encode_mask(mask)
            images.append(img)
            masks.append(mask_encoded)

        # Convert to numpy arrays
        images = np.array(images, dtype=np.float32)
        masks = np.array(masks, dtype=np.uint8)

        # Normalize images and one-hot encode masks
        images = normalize(images, axis=1)
        masks = to_categorical(masks, num_classes=self.n_classes)

        return images, masks

    def label_encode_mask(self, mask):
        """
        Label encode the mask by mapping RGB values to class indices.
        """
        h, w, c = mask.shape
        mask_encoded = np.zeros((h, w), dtype=np.uint8)
        
        # # Debug: Print unique RGB values in the raw mask
        # unique_rgb = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
        # print("Unique RGB values in mask:", unique_rgb)
        
        # Define class RGB values as a dictionary
        class_rgb_values = {
            (0, 0, 0): 0,      # Black (background)
            (0, 255, 0): 1,    # Green
            (0, 255, 255): 2,  # Cyan
            (153, 146, 255): 3,# Purple
            (64, 64, 128): 4,  # Dark Blue
            (255, 255, 0): 5,  # Yellow
            (255, 60, 255): 6, # Magenta
            (255, 55, 55): 7   # Red
        }
        
        # Step 1: Exact matching for background and green (high priority)
        mask_encoded[np.all(mask == (0, 0, 0), axis=-1)] = 0  # Background
        mask_encoded[np.all(mask == (0, 255, 0), axis=-1)] = 1  # Green
        
        # Step 2: Exact matching for other classes
        for rgb, class_idx in class_rgb_values.items():
            if rgb not in [(0, 0, 0), (0, 255, 0)]:  # Skip background and green
                mask_encoded[np.all(mask == rgb, axis=-1)] = class_idx
        
        # Step 3: Tolerance-based matching for green and background if not found
        if 0 not in np.unique(mask_encoded):
            print("Background (class 0) not found, applying tolerance...")
            mask_encoded[np.all(np.abs(mask - (0, 0, 0)) <= 5, axis=-1)] = 0
        if 1 not in np.unique(mask_encoded):
            print("Green (class 1) not found, applying tolerance...")
            mask_encoded[np.all(np.abs(mask - (0, 255, 0)) <= 10, axis=-1)] = 1
        
        # Step 4: Tolerance-based matching for other classes
        tolerance = 5
        for rgb, class_idx in class_rgb_values.items():
            if rgb not in [(0, 0, 0), (0, 255, 0)]:  # Skip background and green
                mask_ = np.all(np.abs(mask - rgb) <= tolerance, axis=-1)
                mask_encoded[mask_] = class_idx
        
        # # Debug: Print unique class labels and their counts
        # unique_labels, counts = np.unique(mask_encoded, return_counts=True)
        # print("Unique class labels after encoding:", dict(zip(unique_labels, counts)))
        
        # Warn if unexpected RGB values remain unclassified
        unclassified_pixels = np.sum(mask_encoded == 0) - np.sum(np.all(mask == (0, 0, 0), axis=-1))
        if unclassified_pixels > 0:
            print(f"Warning: {unclassified_pixels} pixels not matched to any class (excluding background).")
        
        return mask_encoded

# Define the class RGB values
class_rgb_values = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (0, 255, 255),
    3: (153, 146, 255),
    4: (64, 64, 128),
    5: (255, 255, 0),
    6: (255, 60, 255),
    7: (255, 55, 55)
}

class_rgb_array = np.array([(0, 0, 0), (0, 255, 0), (0, 255, 255), (153, 146, 255), 
                            (64, 64, 128), (255, 255, 0), (255, 60, 255), (255, 55, 55)], dtype=np.uint8)
n_classes = len(class_rgb_values)

# Get the list of image and mask file paths
image_paths = glob.glob('F:/Istiak/Dataset/Radiotherapy/Cervix_hdr_axial_flipped_mask/test/images/*png')
mask_paths = glob.glob('F:/Istiak/Dataset/Radiotherapy/Cervix_hdr_axial_flipped_mask/test/masks/*png')

# Sort the paths for matching image-mask pairs
image_paths.sort()
mask_paths.sort()

# Split the dataset into training and testing sets (10% test size)
train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(
    image_paths, mask_paths, test_size=0.10, random_state=0)

# Image size
image_size = (256, 256)

# Create the training and testing generators
train_generator = DataGenerator(train_image_paths, train_mask_paths, batch_size=4, image_size=image_size, class_rgb_array=class_rgb_array, n_classes=n_classes)
test_generator = DataGenerator(test_image_paths, test_mask_paths, batch_size=4, image_size=image_size, class_rgb_array=class_rgb_array, n_classes=n_classes)

# Example usage to get a batch:
train_images, train_masks = train_generator[0]
print("Train batch image shape:", train_images.shape)
print("Train batch mask shape:", train_masks.shape)
train_images, train_masks = train_generator[0]
print("Unique values in the mask (should be class labels 0-7):", np.unique(np.argmax(train_masks, axis=-1)))


import random
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Set the number of images to visualize
num_samples = 1  # Visualize 3 samples to check for green

# Get a batch from the train_generator
train_images, train_masks = train_generator[0]

# Define class RGB values
class_rgb_values = {
    0: (0, 0, 0),      # Black (background)
    1: (0, 255, 0),    # Green
    2: (0, 255, 255),  # Cyan
    3: (153, 146, 255),# Purple
    4: (64, 64, 128),  # Dark Blue
    5: (255, 255, 0),  # Yellow
    6: (255, 60, 255), # Magenta
    7: (255, 55, 55)   # Red
}

# Function to convert class labels to RGB mask
def label_to_rgb(mask, class_rgb_values):
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in class_rgb_values.items():
        rgb_mask[mask == label] = color
    return rgb_mask

# Create a figure for multiple samples
plt.figure(figsize=(15, 5 * num_samples))

for i in range(num_samples):
    # Select a random index from the batch
    random_idx = random.choice(range(len(train_images)))

    # Retrieve the selected image and mask
    image = train_images[random_idx]  # Normalized image (float32, [0, 1])
    mask = train_masks[random_idx]    # One-hot encoded mask (H, W, n_classes)

    # Debug: Print image and mask info
    print(f"Sample {i+1} - Image shape:", image.shape, "min/max:", image.min(), image.max())
    print(f"Sample {i+1} - Mask shape:", mask.shape)

    # Denormalize the image for visualization
    image_uint8 = (image * 255).astype(np.uint8)

    # Convert BGR to RGB for the image
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2RGB)

    # Convert one-hot encoded mask to class labels
    mask_labels = np.argmax(mask, axis=-1).astype(np.uint8)  # Shape: (H, W)
    unique_labels, counts = np.unique(mask_labels, return_counts=True)
    print(f"Sample {i+1} - Unique class labels in mask:", dict(zip(unique_labels, counts)))

    # Convert mask to RGB
    mask_rgb = label_to_rgb(mask_labels, class_rgb_values)
    print(f"Sample {i+1} - Unique RGB values in mask_rgb:", np.unique(mask_rgb.reshape(-1, mask_rgb.shape[2]), axis=0))

    # Overlay with transparency
    overlay = cv2.addWeighted(image_rgb, 0.7, mask_rgb, 0.3, 0)

    # Plot images
    # Original Image
    plt.subplot(num_samples, 3, i * 3 + 1)
    plt.imshow(image_rgb)
    plt.title(f"Sample {i+1} - Original Image")
    plt.axis("off")

    # Mask
    plt.subplot(num_samples, 3, i * 3 + 2)
    plt.imshow(mask_rgb)
    plt.title(f"Sample {i+1} - Mask")
    plt.axis("off")

    # Overlayed Image
    plt.subplot(num_samples, 3, i * 3 + 3)
    plt.imshow(overlay)
    plt.title(f"Sample {i+1} - Overlayed Image")
    plt.axis("off")

plt.tight_layout()




# Example: Get a batch of images and masks from the generator
train_images, train_masks = train_generator[0]

# Now, you can get the height, width, and channels from the train_images
IMG_HEIGHT = train_images.shape[1]
IMG_WIDTH = train_images.shape[2]
IMG_CHANNELS = train_images.shape[3]  # Assuming your images are in RGB format (i.e., 3 channels)





from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose, BatchNormalization, Activation, LeakyReLU, Multiply, Add, GlobalAveragePooling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform

def attention_block(x, g, inter_channel):
    theta_x = Conv2D(inter_channel, (1, 1), strides=(1, 1))(x)
    phi_g = Conv2D(inter_channel, (1, 1), strides=(1, 1))(g)
    f = Activation('relu')(Add()([theta_x, phi_g]))
    psi_f = Conv2D(1, (1, 1), strides=(1, 1))(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = Multiply()([x, rate])
    return att_x

def upsample_block(x, filters, kernel_size=(3, 3), padding='same', strides=1):
    x = Conv2DTranspose(filters, kernel_size, padding=padding, strides=strides)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)  # LeakyReLU activation
    return x

def multi_unet_model2(n_classes=5, IMG_HEIGHT=SIZE_X, IMG_WIDTH=SIZE_Y, IMG_CHANNELS=1):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(s)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c5)

    # Extra layer with 512 channels (bottleneck)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c5)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c6)

    # Expansive path with attention gates
    u6 = upsample_block(c6, 256, strides=(2, 2))
    att6 = attention_block(c4, u6, 128)
    u6 = concatenate([u6, att6])
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u6)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c7)

    u7 = upsample_block(c7, 128, strides=(2, 2))
    att7 = attention_block(c3, u7, 64)
    u7 = concatenate([u7, att7])
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u7)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c8)

    u8 = upsample_block(c8, 64, strides=(2, 2))
    att8 = attention_block(c2, u8, 32)
    u8 = concatenate([u8, att8])
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u8)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c9)

    u9 = upsample_block(c9, 32, strides=(2, 2))
    att9 = attention_block(c1, u9, 16)
    u9 = concatenate([u9, att9])
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(u9)
    c10 = BatchNormalization()(c10)
    c10 = Dropout(0.1)(c10)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=glorot_uniform(), padding='same')(c10)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c10)

    # Create the full segmentation model
    model = Model(inputs=[inputs], outputs=[outputs])

    # Create a secondary model for feature extraction
    feature_extractor = Model(inputs=[inputs], outputs=[c6])  # Extract features from the bottleneck layer

    return model, feature_extractor

segmentation_model, feature_extractor = multi_unet_model2(n_classes, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)






import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from keras.metrics import binary_accuracy



import tensorflow as tf

def focal_loss(y_true, y_pred, gamma=2.0):
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * tf.math.log(y_pred)


    loss = tf.pow(1 - y_pred, gamma) * cross_entropy
    return tf.reduce_mean(loss, axis=-1)


def soft_dice_coefficient(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is float32
    y_pred = tf.cast(y_pred, tf.float32)  # Ensure y_pred is float32
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    sum_true = tf.reduce_sum(y_true, axis=(1, 2, 3))
    sum_pred = tf.reduce_sum(y_pred, axis=(1, 2, 3))
    dice_coefficient = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
    return tf.reduce_mean(dice_coefficient)


def soft_dice_loss(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    sum_true = tf.reduce_sum(y_true, axis=(1, 2, 3))
    sum_pred = tf.reduce_sum(y_pred, axis=(1, 2, 3))
    dice_coefficient = (2. * intersection + smooth) / (sum_true + sum_pred + smooth)
    dice_loss = 1 - dice_coefficient
    return tf.reduce_mean(dice_loss)



def combined_loss(y_true, y_pred, gamma=2.0, alpha=0.5):
    focal = focal_loss(y_true, y_pred, gamma)
    dice = soft_dice_loss(y_true, y_pred)
    return alpha * focal + (1 - alpha) * dice


import tensorflow as tf


class CustomMeanIoU(tf.keras.metrics.MeanIoU):
   
    def __init__(self, num_classes=None, name=None, dtype=None):
        super(CustomMeanIoU, self).__init__(
            num_classes=num_classes, name=name, dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.argmax(y_true, axis=-1)
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)
    
custom_mIoU_metric = CustomMeanIoU(num_classes=5, name='mean_iou')






## Compile the model
segmentation_model.compile(optimizer=Adam(learning_rate=0.001), loss=combined_loss, metrics=['accuracy', soft_dice_coefficient, custom_mIoU_metric])
# Print model summary
segmentation_model.summary()





# Define Early Stopping and Model Checkpoint
early_stopping = EarlyStopping(monitor='val_mean_iou', patience=40, mode='max', restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model1_t1_600Epochs.keras', monitor='val_mean_iou', mode='max', save_best_only=True)

# Train the model
history = segmentation_model.fit(
    train_generator,
    verbose=1,
    epochs=100,
    validation_data=test_generator,
    callbacks=[early_stopping, model_checkpoint],
    shuffle=False
)




# Define Early Stopping to monitor 'val_dice_coefficient' and restore the best weights
early_stopping = EarlyStopping(monitor='val_mean_iou', patience=100, mode='max', restore_best_weights=True)

# Define Model Checkpoint to save the best model based on 'val_dice_coefficient'
model_checkpoint = ModelCheckpoint('saved_model1_t2_resized_v2_Kaggle_200eo_gamma_2_alpha_0.5_B8_MODEL_SAN.keras', monitor='val_mean_iou', mode='max', save_best_only=True)

from keras.callbacks import LearningRateScheduler

# Define the learning rate schedule function
def lr_schedule(epoch):
    initial_lr = 0.001
    return initial_lr * (0.5 ** (epoch // 10))  # Reducing LR by half every 10 epochs

# Create the learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

from keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_mean_iou', factor=0.5, patience=100, min_lr=1e-6)

# Train the model
history = segmentation_model.fit(
    train_generator,
    verbose=1,
    epochs=50,
    validation_data=test_generator,
    callbacks=[early_stopping, model_checkpoint, reduce_lr,lr_scheduler],
    shuffle=False
)





import random
import matplotlib.pyplot as plt
import numpy as np

# Get a random batch index from the generator
batch_index = random.randint(0, len(test_generator) - 1)

# Access the specific batch
test_images, test_masks = test_generator[batch_index]

# Choose a random index from the batch
random_index = random.randint(0, test_images.shape[0] - 1)

# Extract the randomly chosen image and corresponding mask
test_img = test_images[random_index]
ground_truth = test_masks[random_index]

# Predict the mask for the test image using the trained model
predicted_mask = segmentation_model.predict(np.expand_dims(test_img, axis=0))  # Predict for one image

# Convert the predicted mask from one-hot encoding to class labels
predicted_mask_class = np.argmax(predicted_mask, axis=-1)[0]  # Squeeze extra dimensions

# Check the shape of the mask before one-hot encoding
print("Ground truth mask shape (after generator but before one-hot encoding):", ground_truth.shape)
print("Unique values in the mask (should be class labels):", np.unique(np.argmax(ground_truth, axis=-1)))
print("Unique values in the predicted mask (should be class labels):", np.unique(predicted_mask_class))

# Plotting
plt.figure(figsize=(18, 8))

# Plot the original image
plt.subplot(231)
plt.title('Original Image')
plt.imshow(test_img)  # Display the color image (RGB)

# Plot the labeled mask (ground truth)
plt.subplot(232)
plt.title('Labeled Mask (Ground Truth)')

# Convert one-hot encoded mask back to class labels
mask_classes = np.argmax(ground_truth, axis=-1)
plt.imshow(mask_classes, cmap='cividis')

# Plot the predicted mask
plt.subplot(233)
plt.title('Predicted Mask')
plt.imshow(predicted_mask_class, cmap='cividis')

plt.show()




#### UPDATED ###############
import numpy as np
from keras.metrics import MeanIoU

# Number of classes in your segmentation problem
n_classes = 5

# Initialize MeanIoU metric
iou_metric = MeanIoU(num_classes=n_classes)

# Initialize accumulators for per-class IoU calculation
iou_classes = {i: {'intersection': 0, 'union': 0} for i in range(n_classes)}

# Loop through the test generator to get all the batches
for batch_index in range(len(test_generator)):
    # Get a batch of test images and masks
    test_images, test_masks = test_generator[batch_index]

    # Predict the mask for the batch of test images
    y_pred = segmentation_model.predict(test_images)
    
    # Convert predictions to class labels (argmax)
    y_pred_argmax = np.argmax(y_pred, axis=-1)  # Assuming the last dimension is for class probabilities
    
    # Convert the test masks from one-hot encoded to class labels
    y_true = np.argmax(test_masks, axis=-1)
    
    # Update the IoU metric with the true and predicted masks for the current batch
    iou_metric.update_state(y_true, y_pred_argmax)

    # Accumulate intersection and union for each class
    for i in range(n_classes):
        intersection = np.logical_and(y_true == i, y_pred_argmax == i).sum()
        union = np.logical_or(y_true == i, y_pred_argmax == i).sum()
        iou_classes[i]['intersection'] += intersection
        iou_classes[i]['union'] += union

# Print the Mean IoU across all batches
mean_iou = iou_metric.result().numpy()
print("Mean IoU =", mean_iou)

# Calculate final IoU for each class
class_iou = {}
for i in range(n_classes):
    intersection = iou_classes[i]['intersection']
    union = iou_classes[i]['union']
    if union == 0:
        iou = 1.0  # If no ground truth or predicted pixels for this class, set IoU to 1
    else:
        iou = intersection / union
    class_iou[i] = iou

print("IoU for each class:", class_iou)




from sklearn.metrics import f1_score

# Initialize accumulators for per-class Dice Coefficient
dice_classes = {i: {'numerator': 0, 'denominator': 0} for i in range(n_classes)}

# Loop through the test generator to get all the batches
for batch_index in range(len(test_generator)):
    # Get a batch of test images and masks
    test_images, test_masks = test_generator[batch_index]

    # Predict the mask for the batch of test images
    y_pred = segmentation_model.predict(test_images)
    
    # Convert predictions to class labels (argmax)
    y_pred_argmax = np.argmax(y_pred, axis=-1)  # Assuming the last dimension is for class probabilities
    
    # Convert the test masks from one-hot encoded to class labels
    y_true = np.argmax(test_masks, axis=-1)
    
    # Calculate Dice Coefficient for each class
    for i in range(n_classes):
        y_true_class = (y_true == i).astype(int)
        y_pred_class = (y_pred_argmax == i).astype(int)
        
        # Dice Coefficient = 2 * |A ∩ B| / (|A| + |B|)
        numerator = 2 * np.sum(y_true_class * y_pred_class)
        denominator = np.sum(y_true_class) + np.sum(y_pred_class)
        
        dice_classes[i]['numerator'] += numerator
        dice_classes[i]['denominator'] += denominator

# Calculate final Dice Coefficient for each class
class_dice = {}
for i in range(n_classes):
    numerator = dice_classes[i]['numerator']
    denominator = dice_classes[i]['denominator']
    if denominator == 0:
        dice = 1.0  # If no ground truth or predicted pixels for this class, set Dice to 1
    else:
        dice = numerator / denominator
    class_dice[i] = dice

print("Dice Coefficient for each class:", class_dice)
print("Average Dice Coefficient:", np.mean(list(class_dice.values())))





from scipy.ndimage import distance_transform_edt

def calculate_surface_distances(y_true, y_pred):
    """
    Calculate surface distances between the true and predicted masks.
    
    Args:
        y_true: Ground truth mask (2D array of class indices).
        y_pred: Predicted mask (2D array of class indices).
        
    Returns:
        surface_distances: List of surface distances for each class.
    """
    surface_distances = []
    for i in range(n_classes):
        y_true_class = (y_true == i).astype(int)
        y_pred_class = (y_pred == i).astype(int)
        
        # Calculate distance transforms
        dist_true = distance_transform_edt(1 - y_true_class)
        dist_pred = distance_transform_edt(1 - y_pred_class)
        
        # Surface distances
        surface_distances.append((dist_true[y_pred_class == 1].sum() + dist_pred[y_true_class == 1].sum()) /
                                 (y_true_class.sum() + y_pred_class.sum()))
    return surface_distances

# Initialize accumulators for per-class ASD and NSD
asd_classes = {i: 0 for i in range(n_classes)}
nsd_classes = {i: 0 for i in range(n_classes)}

# Loop through the test generator to get all the batches
for batch_index in range(len(test_generator)):
    # Get a batch of test images and masks
    test_images, test_masks = test_generator[batch_index]

    # Predict the mask for the batch of test images
    y_pred = segmentation_model.predict(test_images)
    
    # Convert predictions to class labels (argmax)
    y_pred_argmax = np.argmax(y_pred, axis=-1)  # Assuming the last dimension is for class probabilities
    
    # Convert the test masks from one-hot encoded to class labels
    y_true = np.argmax(test_masks, axis=-1)
    
    # Calculate ASD and NSD for each class
    surface_distances = calculate_surface_distances(y_true, y_pred_argmax)
    for i in range(n_classes):
        asd_classes[i] += surface_distances[i]
        nsd_classes[i] += surface_distances[i] / np.max(surface_distances)  # Normalized Surface Distance

# Calculate final ASD and NSD for each class
class_asd = {i: asd_classes[i] / len(test_generator) for i in range(n_classes)}
class_nsd = {i: nsd_classes[i] / len(test_generator) for i in range(n_classes)}

print("Average Surface Distance (ASD) for each class:", class_asd)
print("Average ASD:", np.mean(list(class_asd.values())))
print("Normalized Surface Distance (NSD) for each class:", class_nsd)
print("Average NSD:", np.mean(list(class_nsd.values())))




import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

# Define the threshold for bounding box detection (if needed)
threshold = 0.2

# Select a random batch from the test generator
batch_index = random.randint(0, len(test_generator) - 1)
test_images, test_masks = test_generator[batch_index]

# Select a random image from the batch
random_index = random.randint(0, test_images.shape[0] - 1)
test_img = test_images[random_index]
ground_truth = test_masks[random_index]

# Normalize or adjust the test image if needed (assuming it's grayscale)
test_img_norm = test_img[:, :, 0]  # Assuming it's a grayscale image

# Make prediction for the selected test image
prediction = segmentation_model.predict(np.expand_dims(test_img, axis=0))  # Add batch dimension
predicted_img = np.argmax(prediction, axis=-1)[0]  # Get the class labels from prediction (remove batch dim)

# Define class names and corresponding colors
class_names = ['Background', 'Calcification', 'Axilla_Findings', 'Tissue', 'Mass']
colors = ['black', 'orange', 'white', 'yellow', 'red']  # Define a color for each class

# Check unique values in masks for debugging
# For ground truth, use np.argmax to convert one-hot encoded masks to class labels
ground_truth_labels = np.argmax(ground_truth, axis=-1)

# For predicted mask, we already have class labels in predicted_img
print("Unique values in the ground truth mask (should be class labels):", np.unique(ground_truth_labels))
print("Unique values in the predicted mask (should be class labels):", np.unique(predicted_img))

# Plotting
plt.figure(figsize=(16, 12))

# Plot the grayscale test image
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img_norm, cmap='gray')

# Plot the ground truth mask
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth_labels, cmap='cividis', extent=(0, ground_truth.shape[1], ground_truth.shape[0], 0))

# Plot the predicted mask
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='cividis')

# Function to add bounding boxes for given masks
def add_bounding_boxes(mask, colors, class_names):
    for class_index in range(1, len(class_names)):  # Skip 'Background' class (index 0)
        # Find contours for the current class in the provided mask
        contours, _ = cv2.findContours(np.uint8(mask == class_index), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours are found for the current class
        if contours:
            # Get the largest contour by area
            contour = max(contours, key=cv2.contourArea)
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            # Choose the color for the current class
            box_color = colors[class_index]
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=box_color, facecolor='none')
            plt.gca().add_patch(rect)
            # Add class name near the bounding box
            class_name = class_names[class_index]
            text_x = x + w + 5  # Offset the text slightly to the right
            text_y = y + h // 2  # Center the text vertically in the box
            plt.text(text_x, text_y, class_name, color=box_color, fontsize=12, verticalalignment='center')

# Add bounding boxes for the ground truth mask
plt.subplot(232)  # Go to the ground truth subplot
add_bounding_boxes(ground_truth_labels, colors, class_names)

# Add bounding boxes for the predicted mask
plt.subplot(233)  # Go to the predicted mask subplot
add_bounding_boxes(predicted_img, colors, class_names)

plt.show()








import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

# Define the threshold for bounding box detection (if needed)
threshold = 0.2

# Select a random batch from the test generator
batch_index = random.randint(0, len(test_generator) - 1)
test_images, test_masks = test_generator[batch_index]

# Select a random image from the batch
random_index = random.randint(0, test_images.shape[0] - 1)
test_img = test_images[random_index]
ground_truth = test_masks[random_index]

# Normalize or adjust the test image if needed (assuming it's grayscale)
test_img_norm = test_img[:, :, 0]  # Assuming it's a grayscale image

# Make prediction for the selected test image
prediction = segmentation_model.predict(np.expand_dims(test_img, axis=0))  # Add batch dimension
predicted_img = np.argmax(prediction, axis=-1)[0]  # Get the class labels from prediction (remove batch dim)

# Define class names and corresponding colors
class_names = ['Background', 'Calcification', 'Axilla_Findings', 'Tissue', 'Mass']
colors = ['black', 'orange', 'white', 'yellow', 'red']  # Define a color for each class

# Check unique values in masks for debugging
# For ground truth, use np.argmax to convert one-hot encoded masks to class labels
ground_truth_labels = np.argmax(ground_truth, axis=-1)

# For predicted mask, we already have class labels in predicted_img
print("Unique values in the ground truth mask (should be class labels):", np.unique(ground_truth_labels))
print("Unique values in the predicted mask (should be class labels):", np.unique(predicted_img))

# Function to calculate IoU for each class
def calculate_iou(y_true, y_pred, num_classes):
    iou_scores = []
    for class_index in range(num_classes):
        # Create binary masks for the current class
        true_class = (y_true == class_index)
        pred_class = (y_pred == class_index)
        
        # Calculate intersection and union
        intersection = np.logical_and(true_class, pred_class).sum()
        union = np.logical_or(true_class, pred_class).sum()
        
        # Avoid division by zero
        if union == 0:
            iou = 0
        else:
            iou = intersection / union
        
        iou_scores.append(iou)
    return iou_scores

# Calculate IoU for each class
iou_scores = calculate_iou(ground_truth_labels, predicted_img, len(class_names))

# Calculate mean IoU (excluding the background class if needed)
mean_iou = np.mean(iou_scores[1:])  # Exclude background (class 0)

# Print IoU for each class and mean IoU
print("\nIoU for each class:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {iou_scores[i]:.4f}")
print(f"\nMean IoU (excluding background): {mean_iou:.4f}")

# Plotting
plt.figure(figsize=(16, 12))

# Plot the grayscale test image
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img_norm, cmap='gray')

# Plot the ground truth mask
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth_labels, cmap='cividis', extent=(0, ground_truth.shape[1], ground_truth.shape[0], 0))

# Plot the predicted mask
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='cividis')

# Function to add bounding boxes for given masks
def add_bounding_boxes(mask, colors, class_names):
    for class_index in range(1, len(class_names)):  # Skip 'Background' class (index 0)
        # Find contours for the current class in the provided mask
        contours, _ = cv2.findContours(np.uint8(mask == class_index), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If contours are found for the current class
        if contours:
            # Get the largest contour by area
            contour = max(contours, key=cv2.contourArea)
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(contour)
            # Choose the color for the current class
            box_color = colors[class_index]
            # Draw bounding box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=box_color, facecolor='none')
            plt.gca().add_patch(rect)
            # Add class name near the bounding box
            class_name = class_names[class_index]
            text_x = x + w + 5  # Offset the text slightly to the right
            text_y = y + h // 2  # Center the text vertically in the box
            plt.text(text_x, text_y, class_name, color=box_color, fontsize=12, verticalalignment='center')

# Add bounding boxes for the ground truth mask
plt.subplot(232)  # Go to the ground truth subplot
add_bounding_boxes(ground_truth_labels, colors, class_names)

# Add bounding boxes for the predicted mask
plt.subplot(233)  # Go to the predicted mask subplot
add_bounding_boxes(predicted_img, colors, class_names)

plt.show()







import matplotlib.pyplot as plt

# Extract metrics from history
train_iou = history.history['mean_iou']
val_iou = history.history['val_mean_iou']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_iou) + 1)

# Plot Mean IoU
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.plot(epochs, train_iou, 'b', label='Training Mean IoU')
plt.plot(epochs, val_iou, 'r', label='Validation Mean IoU')
plt.title('Training and Validation Mean IoU')
plt.xlabel('Epochs')
plt.ylabel('Mean IoU')
plt.legend()
plt.grid()

# Plot Accuracy
plt.subplot(1, 3, 2)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Plot Loss
plt.subplot(1, 3, 3)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



