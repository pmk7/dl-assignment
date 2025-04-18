import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == '__main__':
    config = {
        'train_csv': 'processed_csvs/train.csv',
        'val_csv': 'processed_csvs/val.csv',
        'model_name': 'run17_regression_3x3_relu_128_l2_dropout',
        'task': 'regression',
        'grayscale': True,
        'activation': 'relu',
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
    
    
# Building model with input shape: (128, 128, 1) best regression performance so far
# Epoch 1/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 41s 166ms/step - loss: 686.6074 - mae: 20.8746 - val_loss: 260.4873 - val_mae: 11.9501
# Epoch 2/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 40s 161ms/step - loss: 292.3093 - mae: 12.4192 - val_loss: 227.1406 - val_mae: 11.0626
# Epoch 3/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 41s 167ms/step - loss: 240.0155 - mae: 11.0001 - val_loss: 192.9025 - val_mae: 9.9145
# Epoch 4/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 45s 182ms/step - loss: 193.3996 - mae: 9.6485 - val_loss: 154.4525 - val_mae: 8.1674
# Epoch 5/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 47s 190ms/step - loss: 158.9761 - mae: 8.5705 - val_loss: 145.5702 - val_mae: 7.9770
# Epoch 6/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 46s 187ms/step - loss: 157.6920 - mae: 8.4447 - val_loss: 136.6948 - val_mae: 7.5446
# Epoch 7/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 46s 186ms/step - loss: 141.6574 - mae: 7.8488 - val_loss: 137.8057 - val_mae: 7.5460
# Epoch 8/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 48s 195ms/step - loss: 125.9549 - mae: 7.3280 - val_loss: 136.2061 - val_mae: 7.4790
# Epoch 9/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 49s 199ms/step - loss: 115.2103 - mae: 7.0068 - val_loss: 141.5481 - val_mae: 7.4950
# Epoch 10/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 46s 186ms/step - loss: 110.8528 - mae: 6.8466 - val_loss: 145.5627 - val_mae: 7.5676
# Epoch 11/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 47s 192ms/step - loss: 106.7870 - mae: 6.5561 - val_loss: 126.9354 - val_mae: 7.1022
# Epoch 12/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 44s 178ms/step - loss: 101.5663 - mae: 6.3420 - val_loss: 134.3249 - val_mae: 7.3604
# Epoch 13/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 46s 188ms/step - loss: 95.9695 - mae: 6.1105 - val_loss: 147.3528 - val_mae: 7.5016
# Epoch 14/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 47s 191ms/step - loss: 94.3504 - mae: 5.9974 - val_loss: 138.4456 - val_mae: 7.2437
# Epoch 15/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 48s 193ms/step - loss: 89.0186 - mae: 5.8422 - val_loss: 125.4909 - val_mae: 6.8254
# Epoch 16/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 47s 189ms/step - loss: 83.1727 - mae: 5.5716 - val_loss: 129.3862 - val_mae: 7.0522
# Epoch 17/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 46s 188ms/step - loss: 83.2861 - mae: 5.5267 - val_loss: 123.4293 - val_mae: 6.8290
# Epoch 18/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 45s 183ms/step - loss: 84.5481 - mae: 5.4586 - val_loss: 125.7401 - val_mae: 6.9855
# Epoch 19/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 45s 185ms/step - loss: 78.2371 - mae: 5.2758 - val_loss: 124.8155 - val_mae: 6.8646
# Epoch 20/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 48s 193ms/step - loss: 78.2165 - mae: 5.2537 - val_loss: 126.7216 - val_mae: 6.8644