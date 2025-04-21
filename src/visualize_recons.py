import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from data_loader import load_dataset
from tensorflow.keras.models import load_model

# Paths and Parameters
MODEL_PATH = 'models/autoencoder_run3.keras'
CSV_PATH = 'processed_csvs/block1_autoencoder.csv'
IMG_SIZE = (128, 128)
GRAYSCALE = True

# Load trained autoencoder
model = load_model(MODEL_PATH)

# Load dataset (no shuffling so we can track indices)
ds = load_dataset(
    csv_path=CSV_PATH,
    img_size=IMG_SIZE,
    task='regression',
    grayscale=GRAYSCALE,
    batch_size=1,
    shuffle=False
)

# Extract some examples
n = 5  # number of samples to visualize
originals = []
recons = []

for i, (img, _) in enumerate(ds):
    if i >= n:
        break
    originals.append(img[0].numpy())
    recons.append(model.predict(img)[0])

# Plotting
plt.figure(figsize=(n * 2, 4))
for i in range(n):
    # Original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(originals[i].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(recons[i].squeeze(), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()