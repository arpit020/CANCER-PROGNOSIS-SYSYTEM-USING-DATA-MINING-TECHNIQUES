
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('lungcancer.csv')
X = dataset.iloc[:, [0,5]].values
y = dataset.iloc[:, 7].values




from sklearn.preprocessing import LabelEncoder

 
from sklearn.cross_validation import train_test_split
training_inputs, testing_inputs, training_outputs,testing_outputs= train_test_split(X, y, test_size = 0.2, random_state = 0) 


# Feature Scaling using normalization

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
training_inputs = sc.fit_transform(training_inputs)
testing_inputs = sc.transform(testing_inputs)


if __name__ == '__main__':
    """print ("Tutorial: Training a logistic regression to detect phishing websites")
    train_inputs, train_outputs, test_inputs, test_outputs 
    print ("Training data loaded.)"""
    classifier = lr()
    print ("Logistic regression classifier created.")

    print ("Beginning model training.")
    classifier.fit(training_inputs, training_outputs)
    print ("Model training completed.")

    
    predictions = classifier.predict(testing_inputs)
    print ("Predictions on testing data computed.")
    
    
    
    
from matplotlib.colors import ListedColormap
X_set, y_set = testing_inputs, testing_outputs
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('yellow', 'blue','red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('yellow', 'blue','red','green'))(i), label = j)
plt.title('plotting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
accuracy = 100 * accuracy_score(testing_outputs, predictions)
print ("The accuracy of your logistic regression on testing data is: ")
print(accuracy)
