import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == '__main__':
    config = {
        'train_csv': 'processed_csvs/train.csv',
        'val_csv': 'processed_csvs/val.csv',
        'model_name': 'run19_regression_3x3_tanh_128_l2_dropout',
        'task': 'regression',
        'grayscale': True,
        'activation': 'tanh',
        'kernel_size': 3,
        'depth': 'shallow',
        'pooling': True,
        'skip_connections': False,
        'dense_layers': 2,
        'dropout_rate': 0.3,
        'l2_reg': 0.005,
        'dense_units': [256, 128],
        'img_size': (128, 128)
    }

    train_model(**config)