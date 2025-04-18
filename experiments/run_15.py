import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == '__main__':
    config = {
        'train_csv': 'processed_csvs/train.csv',
        'val_csv': 'processed_csvs/val.csv',
        'model_name': 'run15_fast_tanh_poolingTrue_128_shallow',
        'task': 'classification',
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
        'img_size': (128, 128),
    }

    train_model(**config)
    
    
#     Building model with input shape: (128, 128, 1) run 1
# Epoch 1/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 40s 162ms/step - accuracy: 0.2701 - loss: 4.0421 - val_accuracy: 0.3640 - val_loss: 2.5379
# Epoch 2/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 40s 161ms/step - accuracy: 0.5109 - loss: 2.2042 - val_accuracy: 0.5910 - val_loss: 1.8043
# Epoch 3/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 40s 163ms/step - accuracy: 0.5479 - loss: 1.8412 - val_accuracy: 0.5634 - val_loss: 1.6185
# Epoch 4/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 41s 166ms/step - accuracy: 0.5670 - loss: 1.6140 - val_accuracy: 0.5583 - val_loss: 1.6260
# Epoch 5/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 41s 167ms/step - accuracy: 0.5818 - loss: 1.4559 - val_accuracy: 0.5920 - val_loss: 1.3952
# Epoch 6/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 42s 172ms/step - accuracy: 0.5993 - loss: 1.3721 - val_accuracy: 0.6155 - val_loss: 1.2438
# Epoch 7/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 47s 191ms/step - accuracy: 0.6095 - loss: 1.2625 - val_accuracy: 0.6207 - val_loss: 1.2836
# Epoch 8/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 48s 196ms/step - accuracy: 0.6240 - loss: 1.2343 - val_accuracy: 0.6033 - val_loss: 1.2352
# Epoch 9/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 48s 195ms/step - accuracy: 0.6241 - loss: 1.2273 - val_accuracy: 0.6227 - val_loss: 1.2009
# Epoch 10/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 55s 223ms/step - accuracy: 0.6271 - loss: 1.2064 - val_accuracy: 0.6309 - val_loss: 1.2047
# Epoch 11/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 54s 220ms/step - accuracy: 0.6511 - loss: 1.1571 - val_accuracy: 0.6411 - val_loss: 1.1654
# Epoch 12/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 58s 232ms/step - accuracy: 0.6387 - loss: 1.1672 - val_accuracy: 0.6288 - val_loss: 1.2479
# Epoch 13/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 54s 218ms/step - accuracy: 0.6508 - loss: 1.2220 - val_accuracy: 0.6278 - val_loss: 1.2896
# Epoch 14/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 52s 210ms/step - accuracy: 0.6447 - loss: 1.2776 - val_accuracy: 0.6329 - val_loss: 1.2851
# Epoch 15/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 50s 205ms/step - accuracy: 0.6662 - loss: 1.1502 - val_accuracy: 0.6135 - val_loss: 1.2417
# Epoch 16/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 50s 202ms/step - accuracy: 0.6563 - loss: 1.1728 - val_accuracy: 0.6411 - val_loss: 1.1930