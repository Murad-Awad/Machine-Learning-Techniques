from sklearn import datasets 
iris = datasets.load_iris()
X = iris.data
Y = iris.target
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)
#Splits dataset into two, 75 for train, 75 for test with respective features and target
from sklearn import tree
classifier1=tree.DecisionTreeClassifier()
classifier1.fit(X_train, Y_train)
predictions = classifier1.predict(X_test)
from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, predictions)
#prints accuracy of the program
from sklearn.neighbors import KNeighborsClassifier
#uses another classifier this time
classifier2=KNeighborsClassifier()
classifier2.fit(X_train, Y_train)
predictions = classifier2.predict(X_test)
from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, predictions)
