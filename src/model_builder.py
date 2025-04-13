# src/model_builder.py
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

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
    batch_norm=False,
    l2_reg=0.0,
    task='regression',
    dense_units=128
):
    """
    builds a cnn model with flexible architecture

    Parameters
    ----------
    input_shape : tuple of int
        shape of the input image including channels
    num_outputs : int
        number of output neurons
    activation : str
        activation function to use (e.g., 'relu', 'tanh')
    kernel_size : int
        size of the convolutional kernels
    depth : str
        'shallow' or 'deep' to control the number of conv layers
    pooling : bool
        whether to apply max pooling after conv layers
    skip_connections : bool
        whether to concatenate feature maps in deep networks
    dense_layers : int
        number of dense layers after flattening
    dropout_rate : float
        dropout rate for regularization
    batch_norm : bool
        whether to apply batch normalization
    l2_reg : float
        l2 regularization factor
    task : str
        'regression' or 'classification'

    Returns
    -------
    model : keras.Model
        compiled keras model
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    regularizer = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # Convolutional layers
    if depth == 'shallow':
        x = layers.Conv2D(32, (kernel_size, kernel_size), activation=None, padding='same', kernel_regularizer=regularizer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        if pooling:
            x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, (kernel_size, kernel_size), activation=None, padding='same', kernel_regularizer=regularizer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        if pooling:
            x = layers.MaxPooling2D()(x)

    elif depth == 'deep':
        skips = []
        for filters in [32, 64, 128, 256]:
            conv = layers.Conv2D(filters, (kernel_size, kernel_size), activation=None, padding='same', kernel_regularizer=regularizer)(x)
            if batch_norm:
                conv = layers.BatchNormalization()(conv)
            conv = layers.Activation(activation)(conv)
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
        x = layers.Dense(dense_units, activation=None, kernel_regularizer=regularizer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    # Output layer
    if task == 'regression':
        outputs = layers.Dense(num_outputs)(x)
    elif task == 'classification':
        outputs = layers.Dense(num_outputs, activation='softmax')(x)
    else:
        raise ValueError("Task must be 'regression' or 'classification'")

    model = models.Model(inputs, outputs)
    return model
