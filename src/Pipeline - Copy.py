from scipy.spatial import distance
#implements euclidean distance algorithm
def euc(a,b):
	return distance.euclidean(a,b)
	#building your own classifier based around Nearest Neighbor classifier
class Scrappy():
	def fit(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train=Y_train
	def predict(self, X_test):
		predictions=[]
		for row in X_test:
				label = self.closest(row)
				predictions.append(label)
		return predictions
	def closest(self, row):
		best_dist = euc(row, self.X_train[0])
		best_index = 0
		for i in range(1, len(self.X_train)):
			dist = euc(row, self.X_train[i])
			if dist<best_dist:
				best_dist=dist
				best_index=i
		return self.Y_train[best_index]

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
#from sklearn.neighbors import KNeighborsClassifier
#uses our own custom classifier now
classifier2=Scrappy()
classifier2.fit(X_train, Y_train)
predictions = classifier2.predict(X_test)
from sklearn.metrics import accuracy_score
print accuracy_score(Y_test, predictions)
#Notice how accurate our new classifier is!
