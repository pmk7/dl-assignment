import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == '__main__':
    config = {
        'train_csv': 'processed_csvs/train.csv',
        'val_csv': 'processed_csvs/val.csv',
        'model_name': 'cnn_tanh_5x5_deep_regression',
        'task': 'regression',
        'grayscale': False,
        'activation': 'tanh',
        'kernel_size': 5,
        'depth': 'deep',
        'pooling': True,
        'skip_connections': False,
        'dense_layers': 2,
    }

    train_model(**config)