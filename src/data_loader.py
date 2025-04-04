# src/data_loader.py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os

def parse_image(filename, label, img_size=(128, 128), grayscale=False):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=1 if grayscale else 3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0  # normalize
    return image, label

def load_dataset(csv_path, img_size=(128, 128), task='regression', grayscale=False, batch_size=32, shuffle=True):
    df = pd.read_csv(csv_path)

    filepaths = df['filepath'].values
    if task == 'regression':
        labels = df['age'].values.astype('float32')
    elif task == 'classification':
        le = LabelEncoder()
        labels = le.fit_transform(df['age_group'])
        num_classes = len(le.classes_)
        labels = to_categorical(labels, num_classes=num_classes)
    else:
        raise ValueError("Task must be 'regression' or 'classification'")

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    ds = ds.map(lambda f, l: parse_image(f, l, img_size, grayscale), num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds
