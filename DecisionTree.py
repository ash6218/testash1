from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state = 3)
clf = DecisionTreeClassifier().fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

pp= [[5,3,2,0.5]]
print(clf.predict(pp))