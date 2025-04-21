# transfer_classifier.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import pandas as pd
from data_loader import load_dataset

# Paths and config
ENCODER_MODEL_PATH = 'best_models/autoencoder_model.keras'
CSV_PATH = 'processed_csvs/block2_classification.csv'
MODEL_SAVE_PATH = 'models/transfer_classifier.keras'
IMG_SIZE = (128, 128)
INPUT_SHAPE = (128, 128, 1)
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 20

# Load pretrained autoencoder
autoencoder = tf.keras.models.load_model(ENCODER_MODEL_PATH)
autoencoder.summary() 

def extract_encoder(autoencoder):
    encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=6).output, name="encoder")
    return encoder

def build_transfer_classifier(encoder, num_classes=6):
    encoder.trainable = False  # freeze encoder

    x = layers.Flatten()(encoder.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=encoder.input, outputs=outputs, name="transfer_classifier")
    return model

if __name__ == '__main__':
    # Load and prepare data
    train_ds = load_dataset(CSV_PATH, img_size=IMG_SIZE, task='classification', grayscale=True)

    # Load autoencoder and extract encoder
    autoencoder = load_model(ENCODER_MODEL_PATH)
    encoder = extract_encoder(autoencoder)

    # Build classifier
    model = build_transfer_classifier(encoder, num_classes=NUM_CLASSES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    model.fit(train_ds, epochs=EPOCHS)

    # Save
    model.save(MODEL_SAVE_PATH)
    print(f"Transfer classifier saved to {MODEL_SAVE_PATH}")