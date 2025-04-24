# src/model_builder.py
import tensorflow as tf
from keras import layers, models, regularizers


def build_cnn_model(
    input_shape=(128, 128, 3),
    num_outputs=1,
    activation='relu',
    kernel_size=1,
    depth='deep',
    pooling=True,
    skip_connections=False,
    dense_layers=1,
    dropout_rate=0.5,
    l2_reg=0.0,
    task='regression',
    dense_units=128
):
    inputs = layers.Input(shape=input_shape)
    x = inputs

    regularizer = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # Convolutional layers
    if depth == 'shallow':
        x = layers.Conv2D(32, (kernel_size, kernel_size), activation=activation, padding='same', kernel_regularizer=regularizer)(x)
        if pooling:
            x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, (kernel_size, kernel_size), activation=activation, padding='same', kernel_regularizer=regularizer)(x)
        if pooling:
            x = layers.MaxPooling2D()(x)

        x = layers.Flatten()(x)

    elif depth == 'deep':
        filters = [64, 128, 256, 512]
        for f in filters:
            shortcut = x
            conv = layers.Conv2D(f, (kernel_size, kernel_size), activation=activation, padding='same', kernel_regularizer=regularizer)(x)
            if pooling:
                conv = layers.MaxPooling2D()(conv)

            if skip_connections:
                if shortcut.shape != conv.shape:
                    shortcut = layers.Conv2D(f, (1, 1), padding='same')(shortcut)
                    if pooling:
                        shortcut = layers.MaxPooling2D()(shortcut)
                x = layers.Add()([shortcut, conv])
            else:
                x = conv

        x = layers.GlobalAveragePooling2D()(x)

    if isinstance(dense_units, int):
        dense_units = [dense_units] * dense_layers

    for units in dense_units:
        x = layers.Dense(units, activation=activation, kernel_regularizer=regularizer)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    if task == 'regression':
        outputs = layers.Dense(num_outputs)(x)
    elif task == 'classification':
        outputs = layers.Dense(num_outputs, activation='softmax')(x)
    else:
        raise ValueError("Task must be 'regression' or 'classification'")

    model = models.Model(inputs, outputs)
    return model
