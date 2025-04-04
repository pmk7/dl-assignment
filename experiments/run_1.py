import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model


if __name__ == '__main__':
    config = {
        'train_csv': 'processed_csvs/train.csv',
        'val_csv': 'processed_csvs/val.csv',
        'model_name': 'cnn_relu_3x3_shallow_regression',
        'task': 'regression',
        'grayscale': False,
        'activation': 'relu',
        'kernel_size': 3,
        'depth': 'shallow',
        'pooling': True,
        'skip_connections': False,
        'dense_layers': 1,
    }

    train_model(**config)
