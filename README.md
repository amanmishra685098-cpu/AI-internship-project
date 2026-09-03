# AI Internship Projects — Codect Technologies

Two projects completed as part of the **1-Month Artificial Intelligence Internship**.

## 1. Handwritten Digit Recognizer 🔢

A Convolutional Neural Network (CNN) built with TensorFlow/Keras that classifies handwritten digits (0–9).

**Highlights**
- CNN architecture with Conv2D, BatchNorm, MaxPooling, and Dropout layers
- Trained and evaluated on a labeled digit-image dataset
- Achieves **~99% test accuracy**
- Generates training/validation accuracy & loss curves
- Generates a sample-predictions grid (true label vs. predicted label)
- Saves the trained model to disk (`digit_recognizer_model.keras`)

**Tech stack:** Python, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib

**Run it**
```bash
pip install tensorflow scikit-learn matplotlib numpy
python digit_recognizer.py
```

**Files**
- `digit_recognizer.py` — training & evaluation script
- `training_history.png` — accuracy/loss curves
- `sample_predictions.png` — example predictions on test images

---

## 2. Movie Recommendation System 🎬

A hybrid recommender that suggests movies using two complementary techniques:

- **Content-based filtering** — TF-IDF vectorization of genres + descriptions, ranked by cosine similarity. Answers: *"What's similar to the movie I just liked?"*
- **Collaborative filtering** — item-item similarity computed from a user–movie ratings matrix. Answers: *"What did people with similar taste also enjoy?"* Also supports personalized, per-user recommendations.

**Tech stack:** Python, pandas, scikit-learn, NumPy

**Run it**
```bash
pip install pandas scikit-learn numpy
python movie_recommender.py
```

**Files**
- `movie_recommender.py` — recommender implementation + demo output

**Sample output**
```
Because you liked 'Inception':
  - Interstellar   (similarity: 0.257)
  - The Matrix      (similarity: 0.254)
  ...

Recommended for user_1:
  - Inception   (score: 11.61)
  - The Notebook (score: 11.42)
  ...
```

---

## About
Completed as part of the AI Internship program at **Codect Technologies**.

**Author:** Aman Mishra
**Contact:** amanmishra685098@gmail.com
