import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == '__main__':
    config = {
        'train_csv': 'processed_csvs/train_filtered_female.csv',
        'val_csv': 'processed_csvs/val_filtered_female.csv',
        'model_name': 'run12_fast_filtered_female_relu_gap_128_shallow',
        'task': 'classification',
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
        'img_size': (128, 128),
    }

    train_model(**config)
    
#  Will select this as my best model for classificiation  model 👑 ***
#     Epoch 1/20 first run
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 48s 190ms/step - accuracy: 0.3298 - loss: 3.2034 - val_accuracy: 0.3333 - val_loss: 2.8465
# Epoch 2/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 50s 203ms/step - accuracy: 0.4160 - loss: 2.4597 - val_accuracy: 0.1544 - val_loss: 2.8145
# Epoch 3/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 55s 225ms/step - accuracy: 0.4506 - loss: 2.0179 - val_accuracy: 0.2076 - val_loss: 2.2098
# Epoch 4/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 74s 299ms/step - accuracy: 0.4671 - loss: 1.7859 - val_accuracy: 0.1585 - val_loss: 3.0933
# Epoch 5/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 63s 258ms/step - accuracy: 0.4855 - loss: 1.6061 - val_accuracy: 0.1411 - val_loss: 2.3384
# Epoch 6/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 56s 227ms/step - accuracy: 0.4835 - loss: 1.5347 - val_accuracy: 0.1370 - val_loss: 3.7303
# Epoch 7/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 56s 229ms/step - accuracy: 0.5016 - loss: 1.4466 - val_accuracy: 0.2536 - val_loss: 1.9613
# Epoch 8/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 54s 218ms/step - accuracy: 0.4970 - loss: 1.4354 - val_accuracy: 0.3978 - val_loss: 1.7912
# Epoch 9/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 53s 213ms/step - accuracy: 0.5018 - loss: 1.4143 - val_accuracy: 0.3773 - val_loss: 1.9345
# Epoch 10/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 53s 217ms/step - accuracy: 0.5054 - loss: 1.3810 - val_accuracy: 0.3926 - val_loss: 1.7908
# Epoch 11/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 52s 213ms/step - accuracy: 0.5015 - loss: 1.3788 - val_accuracy: 0.1339 - val_loss: 4.5604
# Epoch 12/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 51s 209ms/step - accuracy: 0.5153 - loss: 1.3402 - val_accuracy: 0.3384 - val_loss: 3.8119
# Epoch 13/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 53s 214ms/step - accuracy: 0.5110 - loss: 1.3477 - val_accuracy: 0.1503 - val_loss: 3.6228
# Epoch 14/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 55s 225ms/step - accuracy: 0.5177 - loss: 1.3218 - val_accuracy: 0.1370 - val_loss: 3.3911
# Epoch 15/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 57s 232ms/step - accuracy: 0.5207 - loss: 1.3118 - val_accuracy: 0.2352 - val_loss: 2.4676

# how it performed on test data:
#     Loading model: best_models/run12_best_classication_fast_relu_gap_128_shallow.keras
# Loading test data from: processed_csvs/test.csv
# Evaluating on test set...
# 31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 36ms/step - accuracy: 0.5682 - loss: 1.2890 
# Evaluation Results:
# Test Loss: 1.2813, Test Accuracy: 58.18%


# Building model with input shape: (128, 128, 1) run 2 without Global average pooling and batch normaalization
# Epoch 1/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 40s 160ms/step - accuracy: 0.3607 - loss: 2.5791 - val_accuracy: 0.5153 - val_loss: 1.5778
# Epoch 2/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 40s 161ms/step - accuracy: 0.5364 - loss: 1.5125 - val_accuracy: 0.5695 - val_loss: 1.3405
# Epoch 3/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 50s 201ms/step - accuracy: 0.5460 - loss: 1.3659 - val_accuracy: 0.5787 - val_loss: 1.3035
# Epoch 4/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 49s 198ms/step - accuracy: 0.5749 - loss: 1.3010 - val_accuracy: 0.6166 - val_loss: 1.2273
# Epoch 5/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 47s 192ms/step - accuracy: 0.6122 - loss: 1.2186 - val_accuracy: 0.6268 - val_loss: 1.1643
# Epoch 6/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 51s 209ms/step - accuracy: 0.6034 - loss: 1.2252 - val_accuracy: 0.6411 - val_loss: 1.1505
# Epoch 7/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 59s 240ms/step - accuracy: 0.6158 - loss: 1.1751 - val_accuracy: 0.6319 - val_loss: 1.1455
# Epoch 8/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 59s 240ms/step - accuracy: 0.6313 - loss: 1.1513 - val_accuracy: 0.6350 - val_loss: 1.1399
# Epoch 9/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 60s 242ms/step - accuracy: 0.6344 - loss: 1.1330 - val_accuracy: 0.6125 - val_loss: 1.2370
# Epoch 10/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 61s 246ms/step - accuracy: 0.6305 - loss: 1.1513 - val_accuracy: 0.6503 - val_loss: 1.0945
# Epoch 11/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 63s 257ms/step - accuracy: 0.6488 - loss: 1.1008 - val_accuracy: 0.6063 - val_loss: 1.1691
# Epoch 12/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 62s 253ms/step - accuracy: 0.6487 - loss: 1.1070 - val_accuracy: 0.6360 - val_loss: 1.1271
# Epoch 13/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 60s 243ms/step - accuracy: 0.6490 - loss: 1.0919 - val_accuracy: 0.6452 - val_loss: 1.1036
# Epoch 14/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 60s 243ms/step - accuracy: 0.6563 - loss: 1.0833 - val_accuracy: 0.6431 - val_loss: 1.1400
# Epoch 15/20
# 245/245 ━━━━━━━━━━━━━━━━━━━━ 50s 201ms/step - accuracy: 0.6591 - loss: 1.0866 - val_accuracy: 0.6401 - val_loss: 1.1213


# Building model with input shape: (128, 128, 1) results for filtered female only
# Epoch 1/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 154ms/step - accuracy: 0.2510 - loss: 3.4190 - val_accuracy: 0.2764 - val_loss: 2.0730
# Epoch 2/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 161ms/step - accuracy: 0.3136 - loss: 2.0111 - val_accuracy: 0.3055 - val_loss: 1.8837
# Epoch 3/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 163ms/step - accuracy: 0.3424 - loss: 1.8731 - val_accuracy: 0.3309 - val_loss: 1.8298
# Epoch 4/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 15s 176ms/step - accuracy: 0.3785 - loss: 1.8005 - val_accuracy: 0.3673 - val_loss: 1.7329
# Epoch 5/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 13s 158ms/step - accuracy: 0.3946 - loss: 1.7454 - val_accuracy: 0.4182 - val_loss: 1.6752
# Epoch 6/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 13s 159ms/step - accuracy: 0.4468 - loss: 1.6476 - val_accuracy: 0.4655 - val_loss: 1.5757
# Epoch 7/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 13s 156ms/step - accuracy: 0.4760 - loss: 1.5603 - val_accuracy: 0.4618 - val_loss: 1.6023
# Epoch 8/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 13s 160ms/step - accuracy: 0.5119 - loss: 1.4671 - val_accuracy: 0.4364 - val_loss: 1.5492
# Epoch 9/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 163ms/step - accuracy: 0.5262 - loss: 1.4194 - val_accuracy: 0.5018 - val_loss: 1.4823
# Epoch 10/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 165ms/step - accuracy: 0.5576 - loss: 1.3469 - val_accuracy: 0.5018 - val_loss: 1.4408
# Epoch 11/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 164ms/step - accuracy: 0.5766 - loss: 1.3034 - val_accuracy: 0.4764 - val_loss: 1.4721
# Epoch 12/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 162ms/step - accuracy: 0.5743 - loss: 1.3187 - val_accuracy: 0.5018 - val_loss: 1.4642
# Epoch 13/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 164ms/step - accuracy: 0.5867 - loss: 1.2822 - val_accuracy: 0.5345 - val_loss: 1.4216
# Epoch 14/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 164ms/step - accuracy: 0.5899 - loss: 1.2440 - val_accuracy: 0.5055 - val_loss: 1.4930
# Epoch 15/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 164ms/step - accuracy: 0.6014 - loss: 1.2370 - val_accuracy: 0.4982 - val_loss: 1.4670
# Epoch 16/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 166ms/step - accuracy: 0.6156 - loss: 1.2318 - val_accuracy: 0.5018 - val_loss: 1.4995
# Epoch 17/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 166ms/step - accuracy: 0.6367 - loss: 1.1899 - val_accuracy: 0.5236 - val_loss: 1.4772
# Epoch 18/20
# 83/83 ━━━━━━━━━━━━━━━━━━━━ 14s 164ms/step - accuracy: 0.6490 - loss: 1.1417 - val_accuracy: 0.5236 - val_loss: 1.4329