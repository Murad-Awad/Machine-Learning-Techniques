import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
dataset = load_iris()
#loads one flower from each type in order to test
test = [0, 50, 100]
#training data simply to see if the program works
train_target = np.delete(dataset.target, test)
train_data = np.delete(dataset.data, test, axis = 0)
#testing data 
test_target=dataset.target[test]
test_data=dataset.data[test]
#training a classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
#prints out the dataset's ids/targets and then prints out the predictions to test out the accuracy
print test_target
print clf.predict(test_data)