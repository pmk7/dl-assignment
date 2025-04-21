import os
from tensorflow.keras.models import load_model
from data_loader import load_dataset

CLASSFICIATION_PATH='best_models/run12_best_classification_fast_relu_gap_128_shallow.keras'
CLASSFICIATION_FILTERED_FEMALE_PATH='best_models/run12_best_classification_filtered_female_relu_gap_128_shallow.keras'
REGRESSION_PATH='best_models/run17_regression_3x3_relu_128_l2_dropout.keras'
AUTOENCODER_PATH='best_models/autoencoder.keras'
TRANSFER_CLASSIFIER_PATH= 'best_models/transfer_classifier.keras'

# Hardcoded paths and configuration
MODEL_PATH = TRANSFER_CLASSIFIER_PATH
TEST_CSV = 'processed_csvs/test.csv'
TASK = 'classification'
GRAYSCALE = True
IMG_SIZE = (128, 128)


def evaluate_model(model_path, test_csv, task='regression', grayscale=False):
    print(f"Loading model: {model_path}")
    model = load_model(model_path)
    
    print(model.summary())
    print(f"Total parameters: {model.count_params():,}")
    
    print(model.get_layer('dense').get_config()['kernel_regularizer'])

    # predictions = model.predict()
    # print(predictions)

    # Load test dataset
    print(f"Loading test data from: {test_csv}")
    test_ds = load_dataset(
        test_csv,
        img_size=IMG_SIZE,
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
    evaluate_model(
        model_path=MODEL_PATH,
        test_csv=TEST_CSV,
        task=TASK,
        grayscale=GRAYSCALE
    )
