# MLModel: A General-Purpose Machine Learning Framework
 # Overview
The MLModel class is a flexible and easy-to-use machine learning framework that can handle both traditional machine learning models and deep learning models. It uses scikit-learn for classical machine learning algorithms and TensorFlow/Keras for deep learning.

# Features
Data Cleaning: Handles missing values and removes duplicates.
Data Preprocessing: Splits data into training and testing sets, and scales features using StandardScaler.
Supports Both Classification and Regression:
Classification Models: Logistic Regression, Decision Tree, Random Forest, SVM
Regression Models: Linear Regression, Decision Tree, Random Forest, SVM
Deep Learning Support: Uses TensorFlow/Keras for building neural networks.
Model Training and Evaluation:
For classification tasks, it provides accuracy, confusion matrix, and a detailed classification report.
For regression tasks, it provides the mean squared error (MSE).
# Installation
Before using the MLModel class, make sure you have the following libraries installed:

pip install pandas scikit-learn tensorflow numpy

Here is how to use the MLModel class in a Python script:



# Import libraries and load dataset
from sklearn.datasets import load_breast_cancer, load_boston
from MLModel import MLModel

# Example for Classification (Logistic Regression)
data = load_breast_cancer(as_frame=True).frame
ml_model_class = MLModel(data, target_column='target', model_type='logistic', task='classification', use_deep_learning=False)
ml_model_class.summarize()

# Example for Deep Learning Classification
deep_learning_model = MLModel(data, target_column='target', model_type='deep_learning', task='classification', use_deep_learning=True)
deep_learning_model.summarize()

# Example for Regression (Deep Learning)
boston_data = load_boston(as_frame=True).frame
deep_learning_reg_model = MLModel(boston_data, target_column='MEDV', model_type='deep_learning', task='regression', use_deep_learning=True)
deep_learning_reg_model.summarize()
Parameters
data: The pandas DataFrame containing your dataset.
target_column: The name of the target variable column in your dataset.
model_type: The type of model to use. Options include 'logistic', 'decision_tree', 'random_forest', 'svm', and 'deep_learning'.
task: The task type, either 'classification' or 'regression'.
use_deep_learning: Set to True to use a deep learning model, otherwise False.
Methods
clean_data()
Cleans the data by handling missing values and removing duplicates.

preprocess_data()
Splits the data into training and testing sets and scales the features using StandardScaler.

initialize_model()
Initializes the machine learning model based on the specified model type and task.

build_deep_learning_model()
Builds a simple deep learning model using TensorFlow/Keras.

train_model()
Trains the model using the training data.

evaluate_model()
Evaluates the model and returns:

Classification: Accuracy, confusion matrix, and classification report.
Regression: Mean squared error.
summarize()
Trains the model and prints the evaluation results.

Example Datasets
Breast Cancer Dataset: Used for demonstrating classification.
Boston Housing Dataset: Used for demonstrating regression.
Note: The load_boston dataset is deprecated and might not be available in newer versions of scikit-learn. Consider using another regression dataset if needed.

Author
Idriss Olivier BADO

License
This project is open-source and free to use. Modify and distribute it as needed.
