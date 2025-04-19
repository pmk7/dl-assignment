import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import train_model

if __name__ == '__main__':
    config = {
        'train_csv': 'processed_csvs/train.csv',
        'val_csv': 'processed_csvs/val.csv',
        'model_name': 'run17_best_regression__3x3_relu_128_l2_dropout',
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
    
    
# Building model with input shape: (128, 128, 1) best regression performance so far 👑 *** 
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



# testing on filtered images with babies (age < 4) removed. result? no effect really
# (venv) philipkeogh@WGS365728373583 Assignment % /Users/philipkeogh/Documents/4_Semester/Deep_Learning/Assignment/venv/bin/python /Users/philipkeogh/Documents/4_Semester/Deep_Learning/Assignment/process_dataset.py
# Filtered CSVs created:
# - train_filtered.csv
# - val_filtered.csv
# - test_filtered.csv
# (venv) philipkeogh@WGS365728373583 Assignment % /Users/philipkeogh/Documents/4_Semester/Deep_Learning/Assignment/venv/bin/python /Users/philipkeogh/Documents/4_Semester/Deep_Learning/Assignment/experiments/run_17.py
# Building model with input shape: (128, 128, 1)
# Epoch 1/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 32s 157ms/step - loss: 647.4764 - mae: 20.2780 - val_loss: 293.2626 - val_mae: 13.1592
# Epoch 2/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 34s 170ms/step - loss: 326.6833 - mae: 14.1020 - val_loss: 235.9531 - val_mae: 12.1447
# Epoch 3/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 32s 158ms/step - loss: 270.5294 - mae: 12.5041 - val_loss: 218.2609 - val_mae: 10.8294
# Epoch 4/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 32s 159ms/step - loss: 218.3136 - mae: 11.1265 - val_loss: 174.9478 - val_mae: 9.6983
# Epoch 5/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 32s 160ms/step - loss: 187.8532 - mae: 10.0243 - val_loss: 189.9565 - val_mae: 9.6877
# Epoch 6/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 34s 169ms/step - loss: 185.8822 - mae: 9.9654 - val_loss: 175.5472 - val_mae: 9.2323
# Epoch 7/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 34s 166ms/step - loss: 159.2712 - mae: 9.1098 - val_loss: 198.5196 - val_mae: 9.8193
# Epoch 8/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 38s 190ms/step - loss: 148.5625 - mae: 8.7255 - val_loss: 162.1275 - val_mae: 8.9985
# Epoch 9/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 37s 184ms/step - loss: 141.1842 - mae: 8.4540 - val_loss: 171.0261 - val_mae: 9.4648
# Epoch 10/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 36s 178ms/step - loss: 133.4938 - mae: 8.0747 - val_loss: 155.5242 - val_mae: 8.8178
# Epoch 11/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 38s 186ms/step - loss: 123.5519 - mae: 7.7309 - val_loss: 149.4360 - val_mae: 8.3581
# Epoch 12/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 40s 199ms/step - loss: 121.6000 - mae: 7.6287 - val_loss: 145.9772 - val_mae: 8.3259
# Epoch 13/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 39s 194ms/step - loss: 107.0633 - mae: 7.0354 - val_loss: 149.4902 - val_mae: 8.2172
# Epoch 14/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 41s 200ms/step - loss: 106.8202 - mae: 7.0477 - val_loss: 150.4304 - val_mae: 8.2425
# Epoch 15/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 39s 194ms/step - loss: 106.0301 - mae: 6.9038 - val_loss: 145.2271 - val_mae: 8.3555
# Epoch 16/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 39s 191ms/step - loss: 109.3238 - mae: 6.9924 - val_loss: 154.6133 - val_mae: 8.3175
# Epoch 17/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 39s 192ms/step - loss: 95.5948 - mae: 6.4863 - val_loss: 157.6571 - val_mae: 8.3996
# Epoch 18/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 44s 217ms/step - loss: 95.1470 - mae: 6.3760 - val_loss: 157.1566 - val_mae: 8.2473
# Epoch 19/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 51s 254ms/step - loss: 91.9611 - mae: 6.1491 - val_loss: 141.7668 - val_mae: 7.9979
# Epoch 20/20
# 201/201 ━━━━━━━━━━━━━━━━━━━━ 53s 261ms/step - loss: 88.4373 - mae: 6.1210 - val_loss: 178.2125 - val_mae: 9.3099