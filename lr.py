import numpy as np
import pandas as pd
from math import ceil

# Load Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Preprocess the dataset
dataset['class'] = pd.Categorical(dataset['class'])
dataset['class'] = dataset['class'].cat.codes

# Define the logistic regression function
def logistic_regression(X, y, num_steps, learning_rate):
  
    intercept = np.ones((X.shape[0], 1))
    X = np.hstack((intercept, X))
    
    weights = np.zeros(X.shape[1])
    
    for step in range(num_steps):
        scores = np.dot(X, weights)
        predictions = 1 / (1 + np.exp(-scores))

        output_error_signal = y - predictions
        gradient = np.dot(X.T, output_error_signal)
        weights += learning_rate * gradient
        
    return weights

# Set the seed for the random number generator
np.random.seed(0)

# Define number of folds
k = 5

# Shuffle the dataset
dataset = dataset.sample(frac=1).reset_index(drop=True)

# Split the dataset into k subsets
subset_size = ceil(len(dataset) / k)
subsets = [dataset[i*subset_size:(i+1)*subset_size] for i in range(k)]

# Get the number of classes
num_classes = len(np.unique(dataset['class'].values))

# Apply K-Fold Cross Validation
for i in range(k):
    # Split the subsets into a training set and a test set
    train = pd.concat(subsets[:i] + subsets[i+1:])
    test = subsets[i]

    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # Apply Logistic Regression for each class (One-vs-all)
    weights_all_classes = []
    for c in range(num_classes):
        binary_y_train = np.where(y_train == c, 1, 0)
        weights_c = logistic_regression(X_train, binary_y_train, num_steps=300000, learning_rate=5e-5)
        weights_all_classes.append(weights_c)

    # Print the coefficients and intercept for each fold and each class
    print('Fold ', i+1)
    for c in range(num_classes):
        print('Class ', c)
        print('Coefficients: ', weights_all_classes[c][1:])
        print('Intercept: ', weights_all_classes[c][0])

    # Predict the test set results for each class and choose the class with the highest score
    final_scores_all_classes = [np.dot(np.hstack((np.ones((X_test.shape[0], 1)), X_test)), weights_c) for weights_c in weights_all_classes]
    preds = np.argmax(final_scores_all_classes, axis=0)

    print('Misclassified samples: %d' % (y_test != preds).sum())

    # Evaluate the model: calculate accuracy, precision, recall and f1-score for each fold
    
    TP = np.sum((y_test == 1) & (preds == 1))
    TN = np.sum((y_test == 0) & (preds == 0))
    FP = np.sum((y_test == 0) & (preds == 1))
    FN = np.sum((y_test == 1) & (preds == 0))
    
    accuracy = (preds == y_test).mean()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)

    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 Score: ', f1_score)