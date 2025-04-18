import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == '__main__':
    config = {
    'train_csv': 'processed_csvs/train.csv',
    'val_csv': 'processed_csvs/val.csv',
    'model_name': 'run11_cnn_classification_leaky_relu_l2_dropout_bn',
    'task': 'classification',
    'grayscale': True,
    'activation': 'leaky_relu',
    'kernel_size': 5,
    'depth': 'deep',
    'pooling': True,
    'skip_connections': False,
    'dense_layers': 3,
    'dropout_rate': 0.3,
    'batch_norm': True,
    'l2_reg': 0.005,
    'dense_units': 512,
    'img_size': (224, 224),
    'use_gap': True
}

    train_model(**config)