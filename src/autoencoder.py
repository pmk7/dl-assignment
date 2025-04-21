# src/autoencoder_run3.py
import tensorflow as tf
from keras import layers, models
import pandas as pd
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping

# Paths
block1_csv = "processed_csvs/block1_autoencoder.csv"
model_save_path = "models/autoencoder_run3.keras"

# Parameters
img_size = (128, 128)
input_shape = (128, 128, 1)
batch_size = 32
epochs = 50



def load_autoencoder_dataset(csv_path, img_size=(224, 224), grayscale=True, batch_size=32, shuffle=True):
    df = pd.read_csv(csv_path)
    filepaths = df['filepath'].values

    def process_image(filepath):
        image = tf.io.read_file(filepath)
        image = tf.image.decode_png(image, channels=1 if grayscale else 3)
        image = tf.image.resize(image, img_size)
        image = image / 255.0
        return image, image  # (input, target)

    ds = tf.data.Dataset.from_tensor_slices(filepaths)
    ds = ds.map(lambda f: process_image(f), num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_autoencoder(input_shape=(128, 128, 1), encoding_dim=64):
    inputs = tf.keras.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D(2, padding='same')(x)

    # Decoder
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(2)(x)
    decoded = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(inputs, decoded, name="autoencoder_run3")
    return autoencoder


if __name__ == "__main__":
    # Load data
    train_ds = load_autoencoder_dataset(block1_csv, img_size=img_size, grayscale=True, batch_size=batch_size)

    # Build and compile model
    model = build_autoencoder(input_shape=input_shape)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    # Train
    model.fit(train_ds, epochs=epochs)

    # Save model
    Path("models").mkdir(exist_ok=True)
    model.save(model_save_path)
    print(f"Autoencoder model saved to {model_save_path}")

    
    
# Epoch 1/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 44s 381ms/step - loss: 0.0319 - mae: 0.1334
# Epoch 2/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 41s 362ms/step - loss: 0.0034 - mae: 0.0397 
# Epoch 3/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 43s 383ms/step - loss: 0.0022 - mae: 0.0317 
# Epoch 4/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 45s 398ms/step - loss: 0.0018 - mae: 0.0279 
# Epoch 5/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 46s 409ms/step - loss: 0.0015 - mae: 0.0259 
# Epoch 6/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 46s 409ms/step - loss: 0.0013 - mae: 0.0237 
# Epoch 7/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 45s 401ms/step - loss: 0.0012 - mae: 0.0231 
# Epoch 8/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 47s 419ms/step - loss: 0.0013 - mae: 0.0245 
# Epoch 9/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 47s 418ms/step - loss: 0.0010 - mae: 0.0211     
# Epoch 10/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 49s 434ms/step - loss: 0.0010 - mae: 0.0212     
# Epoch 11/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 50s 441ms/step - loss: 9.2791e-04 - mae: 0.0198
# Epoch 12/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 53s 471ms/step - loss: 9.1460e-04 - mae: 0.0200 
# Epoch 13/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 48s 427ms/step - loss: 9.3617e-04 - mae: 0.0204
# Epoch 14/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 50s 448ms/step - loss: 8.3912e-04 - mae: 0.0189 
# Epoch 15/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 50s 448ms/step - loss: 8.2607e-04 - mae: 0.0188 
# Epoch 16/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 50s 449ms/step - loss: 7.8619e-04 - mae: 0.0184 
# Epoch 17/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 50s 444ms/step - loss: 7.7354e-04 - mae: 0.0182 
# Epoch 18/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 49s 437ms/step - loss: 7.3919e-04 - mae: 0.0177 
# Epoch 19/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 49s 438ms/step - loss: 7.3024e-04 - mae: 0.0177 
# Epoch 20/20
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 45s 403ms/step - loss: 7.4672e-04 - mae: 0.0181 
# Autoencoder model saved to models/autoencoder_model.keras

# -- second run with decreased batch size (16) and increased filteres (64 -> 128 -> 256)
# (venv) philipkeogh@WGS365728373583 Assignment % /Users/philipkeogh/Documents/4_Semester/Deep_Learning/Assignment/venv/bin/python /Users/philipkeogh/Documents/4_Semester/Deep_Learning/Assignment/src/autoencoder.py
# Epoch 1/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 135s 602ms/step - loss: 0.0200 - mae: 0.0962
# Epoch 2/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 134s 600ms/step - loss: 0.0019 - mae: 0.0294
# Epoch 3/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 132s 593ms/step - loss: 0.0012 - mae: 0.0234
# Epoch 4/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 144s 646ms/step - loss: 0.0010 - mae: 0.0207    
# Epoch 5/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 145s 648ms/step - loss: 8.8127e-04 - mae: 0.0195
# Epoch 6/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 138s 620ms/step - loss: 8.0263e-04 - mae: 0.0189
# Epoch 7/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 144s 643ms/step - loss: 7.0310e-04 - mae: 0.0176
# Epoch 8/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 145s 648ms/step - loss: 6.3888e-04 - mae: 0.0165
# Epoch 9/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 148s 661ms/step - loss: 5.9046e-04 - mae: 0.0160
# Epoch 10/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 152s 680ms/step - loss: 5.5422e-04 - mae: 0.0156
# Epoch 11/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 164s 734ms/step - loss: 5.5314e-04 - mae: 0.0157
# Epoch 12/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 155s 694ms/step - loss: 5.3131e-04 - mae: 0.0155
# Epoch 13/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 145s 649ms/step - loss: 5.8752e-04 - mae: 0.0166
# Epoch 14/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 153s 685ms/step - loss: 4.5725e-04 - mae: 0.0140
# Epoch 15/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 174s 778ms/step - loss: 4.4061e-04 - mae: 0.0137
# Epoch 16/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 172s 770ms/step - loss: 5.6275e-04 - mae: 0.0160
# Epoch 17/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 151s 677ms/step - loss: 4.0622e-04 - mae: 0.0129
# Epoch 18/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 167s 748ms/step - loss: 4.4953e-04 - mae: 0.0143
# Epoch 19/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 184s 823ms/step - loss: 4.0329e-04 - mae: 0.0133
# Epoch 20/20
# 223/223 ━━━━━━━━━━━━━━━━━━━━ 181s 809ms/step - loss: 4.4751e-04 - mae: 0.0143
# Autoencoder model saved to models/autoencoder_model.keras

# same approach as run 1, but increased number of epichs to 50
# Epoch 1/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 50s 435ms/step - loss: 0.0252 - mae: 0.1174
# Epoch 2/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 46s 406ms/step - loss: 0.0031 - mae: 0.0380 
# Epoch 3/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 47s 417ms/step - loss: 0.0020 - mae: 0.0297 
# Epoch 4/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 55s 487ms/step - loss: 0.0017 - mae: 0.0279 
# Epoch 5/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 59s 522ms/step - loss: 0.0013 - mae: 0.0243 
# Epoch 6/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 59s 521ms/step - loss: 0.0012 - mae: 0.0226 
# Epoch 7/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 57s 507ms/step - loss: 0.0011 - mae: 0.0222
# Epoch 8/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 53s 470ms/step - loss: 0.0013 - mae: 0.0239 
# Epoch 9/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 54s 481ms/step - loss: 9.8896e-04 - mae: 0.0205 
# Epoch 10/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 54s 482ms/step - loss: 9.4413e-04 - mae: 0.0201
# Epoch 11/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 55s 493ms/step - loss: 9.2271e-04 - mae: 0.0201 
# Epoch 12/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 57s 505ms/step - loss: 8.8889e-04 - mae: 0.0197 
# Epoch 13/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 64s 565ms/step - loss: 8.3706e-04 - mae: 0.0189 
# Epoch 14/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 62s 550ms/step - loss: 7.9020e-04 - mae: 0.0182 
# Epoch 15/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 63s 556ms/step - loss: 9.2315e-04 - mae: 0.0208 
# Epoch 16/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 64s 571ms/step - loss: 7.3772e-04 - mae: 0.0176 
# Epoch 17/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 58s 518ms/step - loss: 7.2974e-04 - mae: 0.0178 
# Epoch 18/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 54s 482ms/step - loss: 7.0212e-04 - mae: 0.0173 
# Epoch 19/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 58s 517ms/step - loss: 6.8119e-04 - mae: 0.0170 
# Epoch 20/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 55s 489ms/step - loss: 7.0048e-04 - mae: 0.0175 
# Epoch 21/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 61s 539ms/step - loss: 6.6335e-04 - mae: 0.0168 
# Epoch 22/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 59s 526ms/step - loss: 6.4669e-04 - mae: 0.0167 
# Epoch 23/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 54s 481ms/step - loss: 6.8610e-04 - mae: 0.0174 
# Epoch 24/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 58s 512ms/step - loss: 6.0215e-04 - mae: 0.0157 
# Epoch 25/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 56s 501ms/step - loss: 5.9650e-04 - mae: 0.0158 
# Epoch 26/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 55s 485ms/step - loss: 6.1702e-04 - mae: 0.0164 
# Epoch 27/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 58s 513ms/step - loss: 5.8619e-04 - mae: 0.0157 
# Epoch 28/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 58s 513ms/step - loss: 6.6332e-04 - mae: 0.0175 
# Epoch 29/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 58s 518ms/step - loss: 5.9181e-04 - mae: 0.0162 
# Epoch 30/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 57s 503ms/step - loss: 5.6927e-04 - mae: 0.0158 
# Epoch 31/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 55s 489ms/step - loss: 5.3484e-04 - mae: 0.0149 
# Epoch 32/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 58s 517ms/step - loss: 6.3508e-04 - mae: 0.0172 
# Epoch 33/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 60s 535ms/step - loss: 5.3971e-04 - mae: 0.0152 
# Epoch 34/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 58s 517ms/step - loss: 5.3078e-04 - mae: 0.0148 
# Epoch 35/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 55s 484ms/step - loss: 5.1766e-04 - mae: 0.0147 
# Epoch 36/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 57s 506ms/step - loss: 5.3345e-04 - mae: 0.0151 
# Epoch 37/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 56s 496ms/step - loss: 5.0775e-04 - mae: 0.0147 
# Epoch 38/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 64s 572ms/step - loss: 4.9401e-04 - mae: 0.0143 
# Epoch 39/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 78s 690ms/step - loss: 5.9641e-04 - mae: 0.0165
# Epoch 40/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 73s 648ms/step - loss: 4.8603e-04 - mae: 0.0143 
# Epoch 41/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 57s 509ms/step - loss: 4.8919e-04 - mae: 0.0144 
# Epoch 42/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 53s 472ms/step - loss: 4.9580e-04 - mae: 0.0147 
# Epoch 43/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 56s 496ms/step - loss: 5.0712e-04 - mae: 0.0149 
# Epoch 44/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 52s 458ms/step - loss: 4.8405e-04 - mae: 0.0143 
# Epoch 45/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 51s 454ms/step - loss: 4.6233e-04 - mae: 0.0138 
# Epoch 46/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 51s 457ms/step - loss: 4.6434e-04 - mae: 0.0140 
# Epoch 47/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 66s 587ms/step - loss: 5.4065e-04 - mae: 0.0157 
# Epoch 48/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 74s 657ms/step - loss: 4.7546e-04 - mae: 0.0144 
# Epoch 49/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 73s 652ms/step - loss: 4.3623e-04 - mae: 0.0135 
# Epoch 50/50
# 112/112 ━━━━━━━━━━━━━━━━━━━━ 81s 719ms/step - loss: 4.6654e-04 - mae: 0.0142 
# Autoencoder model saved to models/autoencoder_run3.keras