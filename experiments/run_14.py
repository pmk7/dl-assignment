import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == '__main__':
    config = {
        'train_csv': 'processed_csvs/train.csv',
        'val_csv': 'processed_csvs/val.csv',
        'model_name': 'run14_fast_relu_poolingFalse_128_shallow',
        'task': 'classification',
        'grayscale': True,
        'activation': 'relu',
        'kernel_size': 3,
        'depth': 'shallow',
        'pooling': False,
        'skip_connections': False,
        'dense_layers': 2,
        'dropout_rate': 0.3,
        'l2_reg': 0.005,
        'dense_units': [256, 128],
        'img_size': (128, 128),
    }

    train_model(**config)
    
    
# Building model with input shape: (128, 128, 1) aborted due to time constraints, but prelim reuslts promising 
# Epoch 1/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 512s 2s/step - accuracy: 0.3269 - loss: 5.8856 - val_accuracy: 0.5215 - val_loss: 1.9559
# Epoch 2/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 593s 2s/step - accuracy: 0.4770 - loss: 2.0193 - val_accuracy: 0.5368 - val_loss: 1.8588
# Epoch 3/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 508s 2s/step - accuracy: 0.5070 - loss: 1.8879 - val_accuracy: 0.5532 - val_loss: 1.7472