# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 06:54:26 2024

@author: Idriss Olivier BADO
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

class MLModel:
    def __init__(self, data, target_column, model_type='logistic', task='classification', use_deep_learning=False):
        """
        Initialize the MLModel class.

        Parameters:
        data (DataFrame): The pandas DataFrame containing the data.
        target_column (str): The column name for the target variable.
        model_type (str): Type of model ('logistic', 'decision_tree', 'random_forest', 'svm', 'deep_learning', etc.).
        task (str): Task type ('classification' or 'regression'). Default is 'classification'.
        use_deep_learning (bool): Whether to use deep learning (with TensorFlow). Default is False.
        """
        self.data = data
        self.target_column = target_column
        self.model_type = model_type
        self.task = task
        self.use_deep_learning = use_deep_learning
        self.model = None

        # Clean the data
        self.clean_data()

        # Separate features and target
        self.X = self.data.drop(columns=[self.target_column])
        self.y = self.data[self.target_column]

        # Preprocess data (splitting and scaling)
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data()

        # Model initialization based on type
        if not use_deep_learning:
            self.initialize_model()
        else:
            self.build_deep_learning_model()

    def clean_data(self):
        """
        Clean the data by handling missing values and removing duplicates.
        """
        # Fill missing values (you can customize the method based on the dataset)
        self.data.fillna(self.data.mean(), inplace=True)

        # Remove duplicates
        self.data.drop_duplicates(inplace=True)

    def preprocess_data(self):
        """
        Splits the data into training and test sets, and scales the features.

        Returns:
        X_train, X_test, y_train, y_test: Split and scaled datasets.
        """
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

        # Scale features using StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def initialize_model(self):
        """
        Initializes the machine learning model based on the model type and task.
        """
        # Classification models
        if self.task == 'classification':
            if self.model_type == 'logistic':
                self.model = LogisticRegression()
            elif self.model_type == 'decision_tree':
                self.model = DecisionTreeClassifier()
            elif self.model_type == 'random_forest':
                self.model = RandomForestClassifier()
            elif self.model_type == 'svm':
                self.model = SVC()
            else:
                raise ValueError(f"Unsupported classification model type: {self.model_type}")

        # Regression models
        elif self.task == 'regression':
            if self.model_type == 'linear':
                self.model = LinearRegression()
            elif self.model_type == 'decision_tree':
                self.model = DecisionTreeRegressor()
            elif self.model_type == 'random_forest':
                self.model = RandomForestRegressor()
            elif self.model_type == 'svm':
                self.model = SVR()
            else:
                raise ValueError(f"Unsupported regression model type: {self.model_type}")
        else:
            raise ValueError("Unsupported task type. Choose 'classification' or 'regression'.")

    def build_deep_learning_model(self):
        """
        Builds a deep learning model using TensorFlow/Keras.
        """
        input_dim = self.X_train.shape[1]
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_dim=input_dim))
        self.model.add(Dense(32, activation='relu'))
        if self.task == 'classification':
            self.model.add(Dense(1, activation='sigmoid'))  # Binary classification
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        elif self.task == 'regression':
            self.model.add(Dense(1))
            self.model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    def train_model(self):
        """
        Train the machine learning model using the training data.
        """
        if self.use_deep_learning:
            self.model.fit(self.X_train, self.y_train, epochs=50, batch_size=32, verbose=1)
        else:
            self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the machine learning model on the test set.
        
        Returns:
        For classification: accuracy, confusion_matrix, classification_report.
        For regression: mean_squared_error.
        """
        if self.use_deep_learning:
            y_pred = (self.model.predict(self.X_test) > 0.5).astype(int) if self.task == 'classification' else self.model.predict(self.X_test)
        else:
            y_pred = self.model.predict(self.X_test)

        if self.task == 'classification':
            # Classification metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            class_report = classification_report(self.y_test, y_pred)
            return accuracy, conf_matrix, class_report

        elif self.task == 'regression':
            # Regression metrics
            mse = mean_squared_error(self.y_test, y_pred)
            return mse

    def summarize(self):
        """
        Train the model and print evaluation results.
        """
        # Train the model
        self.train_model()

        # Evaluate the model
        if self.task == 'classification':
            accuracy, conf_matrix, class_report = self.evaluate_model()
            print(f"Model Type: {self.model_type.capitalize()} (Classification)")
            print(f"Accuracy: {accuracy:.4f}")
            print("Confusion Matrix:")
            print(conf_matrix)
            print("Classification Report:")
            print(class_report)

        elif self.task == 'regression':
            mse = self.evaluate_model()
            print(f"Model Type: {self.model_type.capitalize()} (Regression)")
            print(f"Mean Squared Error: {mse:.4f}")

# Example Usage:
if __name__ == "__main__":
    # Example dataset (replace with real data)
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True).frame

    # Traditional ML model (Logistic Regression)
    ml_model_class = MLModel(data, target_column='target', model_type='logistic', task='classification', use_deep_learning=False)
    ml_model_class.summarize()

    # Deep Learning example (Binary Classification with TensorFlow)
    deep_learning_model = MLModel(data, target_column='target', model_type='deep_learning', task='classification', use_deep_learning=True)
    deep_learning_model.summarize()

    # Example for regression (using TensorFlow)
    from sklearn.datasets import load_boston
    boston_data = load_boston(as_frame=True).frame

    # Deep Learning Regression
    deep_learning_reg_model = MLModel(boston_data, target_column='MEDV', model_type='deep_learning', task='regression', use_deep_learning=True)
    deep_learning_reg_model.summarize()

