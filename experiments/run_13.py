import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == '__main__':
    config = {
        'train_csv': 'processed_csvs/train.csv',
        'val_csv': 'processed_csvs/val.csv',
        'model_name': 'run13_deep_gap_relu_128',
        'task': 'classification',
        'grayscale': True,
        'activation': 'relu',
        'kernel_size': 3,
        'depth': 'deep',
        'pooling': True,
        'skip_connections': True,       
        'dense_layers': 2,
        'dropout_rate': 0.0,
        'l2_reg': 0.005,
        'dense_units': [128],
        'img_size': (128, 128)
    }

    train_model(**config)