# Breast-Cancer-Classification-DL
This project demonstrates how to perform binary classification using a neural network (with TensorFlow/Keras) to predict breast cancer diagnosis (benign or malignant) based on various features.

Data Exploration and Preprocessing:

    The dataset is loaded and basic statistics are displayed using .info(), .describe(), and .head().
    Irrelevant columns (id, Unnamed: 32) are dropped.
    Missing values are checked, and it's confirmed there are no missing values in the dataset.
    The mean values of features are calculated for each diagnosis group (benign vs. malignant).
    The target variable (diagnosis) is mapped to binary labels: 0 for benign ('B') and 1 for malignant ('M').

Handling Class Imbalance:

    The dataset is imbalanced (with more benign than malignant cases), so the minority class (M) is upsampled to match the majority class (B) using the resample method.

Feature Scaling:

    The features (X) are standardized using StandardScaler to ensure they are on a similar scale, which is important for neural networks.

Modeling with TensorFlow/Keras:

    A simple neural network is built using Keras' Sequential API.
    Flatten: Flattens the input data.
    Dense layers: Two hidden layers with ReLU activations (64 and 32 units).
    Dropout: Dropout layer with a rate of 0.4 to prevent overfitting.
    Output layer: A final dense layer with 2 output units and a sigmoid activation function for binary classification.
    The model is compiled with the Adam optimizer and SparseCategoricalCrossentropy loss, which is suitable for multi-class classification where the labels are integers.
    The model is trained for 10 epochs with a batch size of 32 and uses a 10% validation split to monitor overfitting during training.

Model Evaluation:

    After training, the model is evaluated on the test set, and accuracy is printed.
    Predictions are made on the test set, and the accuracy score and classification report (including precision, recall, and F1 score) are printed.

Key Output:

    Accuracy: The overall accuracy of the model on the test set.
    Classification Report: Detailed metrics (precision, recall, F1-score) for both classes (benign and malignant).

This code provides a complete pipeline for binary classification, from data exploration and preprocessing to model training and evaluation.
