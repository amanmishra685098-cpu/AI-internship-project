"""
Handwritten Digit Recognizer
-----------------------------
Trains a Convolutional Neural Network (CNN) to recognize handwritten digits.

Dataset: scikit-learn's built-in `digits` dataset (1,797 images of handwritten
digits, 0-9, 8x8 pixels each). This dataset ships with scikit-learn, so the
script runs fully offline with no external downloads required.

(Note: if you have internet access and want the full classic MNIST dataset
(60,000 28x28 images) instead, swap the data-loading section for
`tensorflow.keras.datasets.mnist.load_data()` — the rest of the pipeline
below works unchanged either way.)

Requirements:
    pip install tensorflow scikit-learn matplotlib numpy

Run:
    python digit_recognizer.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


def load_data():
    """Load and preprocess the digits dataset."""
    digits = load_digits()
    X = digits.images.astype("float32")
    y = digits.target

    # Normalize pixel values to [0, 1]
    X = X / 16.0  # sklearn digits pixel values range 0-16

    # Add channel dimension: (N, 8, 8) -> (N, 8, 8, 1)
    X = np.expand_dims(X, axis=-1)

    # One-hot encode labels
    y_cat = to_categorical(y, num_classes=10)

    return train_test_split(X, y_cat, y, test_size=0.2, random_state=42, stratify=y)


def build_model(input_shape=(8, 8, 1), num_classes=10):
    """Build a simple CNN for digit classification."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, out_path="training_history.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curves to {out_path}")


def plot_sample_predictions(model, X_test, y_test_labels, out_path="sample_predictions.png", n=10):
    idx = np.random.choice(len(X_test), n, replace=False)
    preds = np.argmax(model.predict(X_test[idx], verbose=0), axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[idx[i]].squeeze(), cmap="gray")
        true_label = y_test_labels[idx[i]]
        pred_label = preds[i]
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {true_label} / Pred: {pred_label}", color=color, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved sample predictions to {out_path}")


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = load_data()
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    print("\nBuilding model...")
    model = build_model()
    model.summary()

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=25,
        batch_size=32,
        verbose=2,
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    plot_history(history)
    plot_sample_predictions(model, X_test, y_test_labels)

    model.save("digit_recognizer_model.keras")
    print("\nModel saved to digit_recognizer_model.keras")


if __name__ == "__main__":
    main()
