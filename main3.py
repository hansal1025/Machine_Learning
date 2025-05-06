# =============================================
#                    Import
# =============================================
import os, random, shutil, sys
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ================================================
#                  GPU Activation
# ================================================
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ================================================
#                  User Setting
# ================================================
# Define your paths
MAIN_DIR = r'Medicinal plant dataset'  # your dataset with class folders
BASE_OUTPUT = 'data_split'
TRAIN_DIR = os.path.join(BASE_OUTPUT, 'train')
VAL_DIR = os.path.join(BASE_OUTPUT, 'val')
TEST_DIR = os.path.join(BASE_OUTPUT, 'test')
model_path = "best_model.keras"

# Define split ratios
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15  # not strictly needed but for clarity

# Parameters
Batch_Size = 32
Epochs = 100
LR = 0.001
patience_stop = 10
patience_reduce = 3
reduction_factor = 0.5

# Image Parameters
IMG_Size = (128, 128)
Channels = 3 # as the dataset has only grayscale images we use 1 or else we must use 3 for rgb images
Input_Shape = IMG_Size + (Channels,)

print(f"\n _________________________________________________________________ \n Printing Parameters for future reference..., \n Main Directory: {MAIN_DIR },\n Best Model: {model_path},\n Training Split: {0.7 * 100:.2f},\n Validation split: {0.15 * 100:.2f},\n Test split: {0.15 * 100:.2f},\n Batch Size: {Batch_Size},\n Epochs: {Epochs},\n Learning Rate: {LR},\n Image Size: {IMG_Size},\n Channels: {Channels},\n Thank You! Lets start with the model...\n _________________________________________________________________")
# ================================================
#                  Data Importing
# ================================================
def prepare_dataset():
    try:
        # 1.1 Ensure destination splits exist
        for split in (TRAIN_DIR, VAL_DIR, TEST_DIR):
            os.makedirs(split, exist_ok=True)

        for cls in tqdm(os.listdir(MAIN_DIR), desc="Classes"):
            cls_src = os.path.join(MAIN_DIR, cls)
            if not os.path.isdir(cls_src) or cls.startswith('.') or 'checkpoint' in cls.lower():
                continue

            # If subfolder Train/Val/Test already exist inside class folder
            subfolder = os.listdir(cls_src)
            if all(s.lower() in [sf.lower() for sf in subfolder] for s in ['train', 'val', 'test']):
                tqdm.write(f"Copying existing splits for {cls}")
                for split_name in ['train', 'val', 'test']:
                    src_split = os.path.join(cls_src, split_name)
                    dst_split = os.path.join(BASE_OUTPUT, split_name, cls)
                    os.makedirs(dst_split, exist_ok=True)
                    for fname in tqdm(os.listdir(src_split),
                                      desc=f"Copying {cls}/{split_name}",
                                      leave=False):
                        src = os.path.join(src_split, fname)
                        dst = os.path.join(dst_split, fname)
                        if os.path.isfile(src) and not os.path.exists(dst):
                            shutil.copy2(src, dst)
            else:
                # If no subfolder, split raw images
                imgs = [f for f in os.listdir(cls_src)
                        if f.lower().endswith(('.jpg', 'jpeg', 'png'))
                        and os.path.isfile(os.path.join(cls_src, f))]
                if not imgs:
                    tqdm.write(f"No images found in {cls_src}")
                    continue

                random.shuffle(imgs)
                n = len(imgs)
                n_train = int(n * TRAIN_SPLIT)
                n_val = int(n * VAL_SPLIT)

                splits = {
                    TRAIN_DIR: imgs[:n_train],
                    VAL_DIR: imgs[n_train:n_train + n_val],
                    TEST_DIR: imgs[n_train + n_val:]
                }

                tqdm.write(f"\nSplitting raw images for {cls} ({n} total)…")
                for split_dir, file_list in splits.items():
                    dst_class = os.path.join(split_dir, cls)
                    os.makedirs(dst_class, exist_ok=True)
                    for fname in tqdm(file_list,
                                      desc=f"{cls} → {os.path.basename(split_dir)}",
                                      leave=False):
                        src = os.path.join(cls_src, fname)
                        dst = os.path.join(dst_class, fname)
                        if os.path.isfile(src) and not os.path.exists(dst):
                            shutil.copy2(src, dst)

    except KeyboardInterrupt:
        print("🚨 File copying interrupted by user.")
        sys.exit(1)

# Run it
if os.path.exists(BASE_OUTPUT):
    print("Already split dataset exists hence dataset creation is skipped.")
else:
    print("🔧 Preparing dataset…")
    prepare_dataset()
    print("Dataset is ready in Train/Val/Test structure!")

# ================================================
#                Data Processing
# ================================================
training_data = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True, # This is false because the human face wont become upside down
    fill_mode = "nearest",
)

train_gen = training_data.flow_from_directory(
    TRAIN_DIR,
    target_size = IMG_Size,
    batch_size = Batch_Size,
    color_mode = "grayscale" if Channels == 1 else "rgb",
    class_mode = "categorical"
)

val_data = ImageDataGenerator(rescale=1./255)
val_gen = val_data.flow_from_directory(
    VAL_DIR,
    target_size=IMG_Size,
    batch_size=Batch_Size,
    color_mode='grayscale' if Channels == 1 else 'rgb',
    class_mode='categorical'
)

test_data = ImageDataGenerator(rescale=1./255)
test_gen = test_data.flow_from_directory(
    TEST_DIR,
    target_size=IMG_Size,
    batch_size=Batch_Size,
    color_mode='grayscale' if Channels == 1 else 'rgb',
    class_mode='categorical',
    shuffle=False # Must be false for testing so that the file names and the predictions match
)

with open('class_indices.json', 'w') as f:
    import json
    json.dump(train_gen.class_indices, f)

# ================================================
#        CNN Model  - Part 1 | Architecture
# ================================================
def build_cnn(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, (2,2), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(64, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(128, (2,2), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)

    x = keras.layers.Conv2D(256, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Dropout(0.15)(x)

    x = keras.layers.Conv2D(512, (3,3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Dropout(0.25)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.25)(x)

    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, output)

    return model

num_classes = train_gen.num_classes

model = build_cnn(Input_Shape, num_classes)
model.summary()

# ================================================
#        CNN Model  - Part 2 | Processing
# ================================================
optim = keras.optimizers.Adam(learning_rate=LR)
loss = loss = keras.losses.CategoricalCrossentropy(from_logits=False)
metrics = ["accuracy"]

model.compile(optimizer = optim, loss = loss, metrics = metrics)

ModelCheckpoint(
    filepath='best_model.keras',  # .keras or .h5 is fine for full model
    monitor='val_loss',
    save_best_only=True,
    verbose=1,
    save_weights_only=False  # <--- this is the key change!
)

callbacks_list = [
    EarlyStopping(monitor="val_loss", patience=patience_stop, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath="best_model.keras", monitor="val_loss", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=reduction_factor, patience=patience_reduce, verbose=1)
]

# ================================================
#      CNN Model  - Part 3 | Class Weights
# ================================================
class_indices = train_gen.class_indices
classes = list(class_indices.keys())

# Get labels from generator
labels = train_gen.classes

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(class_weights))

# ================================================
#   CNN Model  - Part 4 | Final Model Training
# ================================================
if os.path.exists(model_path):
    print("Already trained model exists hence training is skipped. Loading of model taking place ... \n _________________________________________________________________")
    model = keras.models.load_model(model_path)
    print(" _________________________________________________________________ \n Model Loaded \n _________________________________________________________________")
else:
    print("No trained model was found. Training new model in progress... \n _________________________________________________________________")
    history = model.fit(
        train_gen,
        epochs = Epochs,
        validation_data = val_gen,
        callbacks = callbacks_list,
        class_weight=class_weights,
        verbose = 1
    )
    print("_________________________________________________________________\n Model Trained \n _________________________________________________________________")

# ================================================
#         Training Curve Plotting
# ================================================
def plot_training_curves(history):
    # Set plot style
    sns.set_style("whitegrid")

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Loss
    axs[0].plot(history.history['loss'], label='Train Loss', color='blue')
    axs[0].plot(history.history['val_loss'], label='Validation Loss', color='orange')
    axs[0].set_title('Loss Curve', fontsize=16)
    axs[0].set_xlabel('Epochs', fontsize=14)
    axs[0].set_ylabel('Loss', fontsize=14)
    axs[0].legend()
    axs[0].grid(True)

    # Plot Accuracy
    axs[1].plot(history.history['accuracy'], label='Train Accuracy', color='green')
    axs[1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    axs[1].set_title('Accuracy Curve', fontsize=16)
    axs[1].set_xlabel('Epochs', fontsize=14)
    axs[1].set_ylabel('Accuracy', fontsize=14)
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Call this after model.fit()
plot_training_curves(history)


# ================================================
#               Evalute on Test Data
# ================================================
test_loss, test_acc = model.evaluate(test_gen, verbose = 2)
print(f"\n Overall Test Accuracy: {test_acc * 100:.2f}% | Overall Test Loss: {test_loss:.4f}")

# ================================================
#                 Results Analysis
# ================================================
# Predict on test set
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# True labels from test generator
y_true = test_gen.classes

# Get corresponding filenames (without directory info)
filenames = test_gen.filenames
# Map indices to class names
class_labels = list(test_gen.class_indices.keys())

# Create a DataFrame logging filenames, true labels, predicted labels, and correctness
results_df = pd.DataFrame({
    'Filename': filenames,
    'True_Label': [class_labels[i] for i in y_true],
    'Predicted_Label': [class_labels[i] for i in y_pred]
})

results_df['Correct'] = results_df['True_Label'] == results_df['Predicted_Label']

# Save CSV file
csv_filename = 'test_predictions_log.csv'
results_df.to_csv(csv_filename, index=False)
print(f"✅ Predictions log saved to {csv_filename}")

# ================================================
#                Results Report
# ================================================

report = classification_report(y_true, y_pred, target_names=class_labels)
print("Classification Report:")
print(report)

# ================================================
#              Confusion Matrix
# ================================================

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.show()