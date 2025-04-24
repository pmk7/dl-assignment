import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
import pandas as pd
from data_loader import load_dataset

TRAIN_CSV = "processed_csvs/block2_classification.csv"
VAL_CSV = "processed_csvs/val_filtered.csv"
MODEL_SAVE_PATH = "models/resnet50_labelled_finetuned_classifier.keras"

# Parameters
IMG_SIZE = (224, 224) 
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS_INITIAL = 5
EPOCHS_FINETUNE = 10

# Load data
train_ds = load_dataset(TRAIN_CSV, img_size=IMG_SIZE, task="classification", grayscale=False, batch_size=BATCH_SIZE, shuffle=True)
val_ds = load_dataset(VAL_CSV, img_size=IMG_SIZE, task="classification", grayscale=False, batch_size=BATCH_SIZE, shuffle=False)

# Load ResNet50 without top
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False 

# Add custom classifier
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = models.Model(inputs, outputs)

# Compile and train (only top layers)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Starting initial training (frozen base)...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_INITIAL)

# Fine-tune: unfreeze base_model
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print("Starting fine-tuning...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE)

# Save model
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")