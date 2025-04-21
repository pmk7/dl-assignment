# src/data_loader.py
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import os

def parse_image(filename, label, img_size=(128, 128), grayscale=False):
    """
    parses and preprocesses an image from a filepath

    Parameters
    ----------
    filename : str
        path to the image file
    label : float or array
        label associated with the image
    img_size : tuple of int
        desired image size as (height, width)
    grayscale : bool
        whether to load image in grayscale mode

    Returns
    -------
    image : tensor
        normalized and resized image tensor
    label : float or array
        unchanged label
    """
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=1 if grayscale else 3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0  # normalize
    return image, label

def load_dataset(csv_path, img_size=(128, 128), task='regression', grayscale=False, batch_size=32, shuffle=True):
    """
    loads a dataset from csv with image paths and labels, with default values which can be overwridden when configuring model

    Parameters
    ----------
    csv_path : str
        path to the csv file containing 'filepath' and labels
    img_size : tuple of int
        desired image size as (height, width)
    task : str
        either 'regression' or 'classification'
    grayscale : bool
        whether to load images in grayscale
    batch_size : int
        number of samples per batch
    shuffle : bool
        whether to shuffle the dataset

    Returns
    -------
    ds : tf.data.Dataset
        prepared tensorflow dataset ready for training or evaluation
    """
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
