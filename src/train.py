# src/train.py
import os
import argparse
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .model_builder import build_cnn_model
from .data_loader import load_dataset
import tensorflow as tf

def train_model(
    train_csv,
    val_csv,
    model_name='cnn_regression',
    save_dir='models/',
    img_size=(128, 128),
    grayscale=False,
    task='regression',
    **kwargs
):
    os.makedirs(save_dir, exist_ok=True)

    # Load datasets
    train_ds = load_dataset(train_csv, img_size, task, grayscale)
    val_ds = load_dataset(val_csv, img_size, task, grayscale, shuffle=False)

    # Determine output units
    num_outputs = 1 if task == 'regression' else kwargs.get('num_classes', 6)

    # Build model
    model = build_cnn_model(
        input_shape=(img_size[0], img_size[1], 1 if grayscale else 3),
        num_outputs=num_outputs,
        task=task,
        **kwargs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mse' if task == 'regression' else 'categorical_crossentropy',
        metrics=['mae'] if task == 'regression' else ['accuracy']
    )

    # Callbacks
    checkpoint_path = os.path.join(save_dir, model_name + '.keras')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks
    )

    return model, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='cnn_regression')
    parser.add_argument('--task', type=str, choices=['regression', 'classification'], default='regression')
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument('--depth', type=str, default='shallow')
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--pooling', action='store_true')
    parser.add_argument('--skip_connections', action='store_true')
    parser.add_argument('--dense_layers', type=int, default=1)
    args = parser.parse_args()

    train_model(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        model_name=args.model_name,
        grayscale=args.grayscale,
        task=args.task,
        activation=args.activation,
        kernel_size=args.kernel_size,
        depth=args.depth,
        pooling=args.pooling,
        skip_connections=args.skip_connections,
        dense_layers=args.dense_layers
    )
