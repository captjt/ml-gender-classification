from sklearn import tree, neighbors, cross_validation, metrics, linear_model, naive_bayes
from matplotlib import pyplot as plt
from utils import show_confusion_matrix

# Original tree decision tree
treeClf = tree.DecisionTreeClassifier()

# The challenge of using 3 more ...
# 1 K Neighbors classification (n = 3)
neighborsClf = neighbors.KNeighborsClassifier(n_neighbors=3)

#2 Logistic regression classification
logClf = linear_model.LogisticRegression()

#3 Naive bayes classification
nbClf = naive_bayes.GaussianNB()


labels = ['Male', 'Female']


# [height, weight, shoe size] -- dataset to train/test
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']


# Training and testing splits
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.3)


# Train our classication models
treeClf = treeClf.fit(X_train, Y_train)
neighborsClf = neighborsClf.fit(X_train, Y_train)
logClf = logClf.fit(X_train, Y_train)
nbClf = nbClf.fit(X_train, Y_train)


# Accuracies
treeAccuracy = treeClf.score(X_test, Y_test)
neighborsAccuracy = neighborsClf.score(X_test, Y_test)
logAccuracy = logClf.score(X_test, Y_test)
nbAccuracy = nbClf.score(X_test, Y_test)


# Predictions
treePrediction = treeClf.predict(X_test)
neighborsPrediction = neighborsClf.predict(X_test)
logPrediction = logClf.predict(X_test)
nbPrediction = nbClf.predict(X_test)


# Confusion matrices
treeCm = metrics.confusion_matrix(treePrediction, Y_test)
neighborsCm = metrics.confusion_matrix(neighborsPrediction, Y_test)
logCm = metrics.confusion_matrix(logPrediction, Y_test)
nbCm = metrics.confusion_matrix(nbPrediction, Y_test)


# Matplotlib Images
show_confusion_matrix('Tree', treeCm, labels)
show_confusion_matrix('K-Neighbors', neighborsCm, labels)
show_confusion_matrix('Logistic Regression', logCm, labels)
show_confusion_matrix('Naive Bayes', nbCm, labels)


# Terminal Outputs ...
# Accuracies
print('Tree accuracy: ', treeAccuracy)
print('Neighbors accuracy: ', neighborsAccuracy)
print('Logistic regression accuracy: ', logAccuracy)
print('Naive bayes accuracy: ', nbAccuracy)

# Predicitons
print('Tree predicition: ', treePrediction)
print('Neighbor prediction: ', neighborsPrediction)
print('Logistic regression prediction: ', logPrediction)
print('Naive bayes prediction: ', nbPrediction)

# Confusion matrices
print('Tree confusion matrices: \n', treeCm)
print('Neighbors confusion matrices: \n', neighborsCm)
print('Logistic regression confusion matrices: \n', logCm)
print('Naive bayes confusion matrices: \n', nbCm)
