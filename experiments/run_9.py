import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == '__main__':
    config = {
        'train_csv': 'processed_csvs/train.csv',
        'val_csv': 'processed_csvs/val.csv',
        'model_name': 'cnn_regression_deep_grayscale_dense256_run9',
        'task': 'regression',
        'grayscale': True,
        'activation': 'relu',
        'kernel_size': 3,
        'depth': 'deep',                    
        'pooling': True,
        'skip_connections': False,        
        'dense_layers': 2,
        'dense_units': 256,
        'dropout_rate': 0.4,
        'batch_norm': True,
        'l2_reg': 0.01
    }

    train_model(**config)