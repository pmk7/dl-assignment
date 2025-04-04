# src/model_builder.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(
    input_shape=(128, 128, 3),
    num_outputs=1,
    activation='relu',
    kernel_size=3,
    depth='shallow',
    pooling=True,
    skip_connections=False,
    dense_layers=1,
    task='regression'
):
    """
    Builds a CNN model with flexible architecture.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Convolutional layers
    if depth == 'shallow':
        x = layers.Conv2D(32, (kernel_size, kernel_size), activation=activation, padding='same')(x)
        if pooling:
            x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, (kernel_size, kernel_size), activation=activation, padding='same')(x)
        if pooling:
            x = layers.MaxPooling2D()(x)

    elif depth == 'deep':
        skips = []
        for filters in [32, 64, 128, 256]:
            conv = layers.Conv2D(filters, (kernel_size, kernel_size), activation=activation, padding='same')(x)
            if pooling:
                conv = layers.MaxPooling2D()(conv)
            if skip_connections:
                skips.append(conv)
            x = conv
        if skip_connections and skips:
            x = layers.Concatenate()(skips)

    x = layers.Flatten()(x)

    # Fully connected layers
    for _ in range(dense_layers):
        x = layers.Dense(128, activation=activation)(x)

    # Output layer
    if task == 'regression':
        outputs = layers.Dense(num_outputs)(x)
    elif task == 'classification':
        outputs = layers.Dense(num_outputs, activation='softmax')(x)
    else:
        raise ValueError("Task must be 'regression' or 'classification'")

    model = models.Model(inputs, outputs)
    return model
