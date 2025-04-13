import argparse
import os
from tensorflow.keras.models import load_model
from data_loader import load_dataset


def evaluate_model(model_path, test_csv, task='regression', grayscale=False):
    # Load model
    print(f"Loading model: {model_path}")
    model = load_model(model_path)

    # Load test dataset
    print(f"Loading test data from: {test_csv}")
    test_ds = load_dataset(
        test_csv,
        img_size=(128, 128),
        task=task,
        grayscale=grayscale,
        batch_size=32,
        shuffle=False
    )

    # Evaluate model
    print("Evaluating on test set...")
    results = model.evaluate(test_ds)
    print("Evaluation Results:")
    if task == 'regression':
        print(f"Test MSE: {results[0]:.4f}, Test MAE: {results[1]:.4f}")
    else:
        print(f"Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]*100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_csv', type=str, default='processed_csvs/test.csv')
    parser.add_argument('--task', type=str, choices=['regression', 'classification'], default='regression')
    parser.add_argument('--grayscale', action='store_true')
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        test_csv=args.test_csv,
        task=args.task,
        grayscale=args.grayscale
    )
