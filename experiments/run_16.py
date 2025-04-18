import sys
import oswhat
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == '__main__':
    config = {
        'train_csv': 'processed_csvs/train.csv',
        'val_csv': 'processed_csvs/val.csv',
        'model_name': 'run16_regression_3x3_tanh_l2_dropout',
        'task': 'regression',
        'grayscale': True,
        'activation': 'tanh',
        'kernel_size': 5,
        'depth': 'shallow',
        'pooling': True,
        'skip_connections': False,
        'dense_layers': 2,
        'dropout_rate': 0.3,
        'l2_reg': 0.005,
        'dense_units': [256, 128],
        'img_size': (224, 224)

    }

    train_model(**config)
    
#     Building model with input shape: (224, 224, 1) aborted
# Epoch 1/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 212s 863ms/step - loss: 807.1476 - mae: 21.6605 - val_loss: 629.0231 - val_mae: 20.8556
# Epoch 2/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 263s 1s/step - loss: 616.4395 - mae: 20.7595 - val_loss: 625.7061 - val_mae: 21.0838
# Epoch 3/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 264s 1s/step - loss: 612.2505 - mae: 20.8180 - val_loss: 624.1892 - val_mae: 21.0863
# Epoch 4/20