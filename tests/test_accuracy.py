from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load and split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# train
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

# accuracy test to ensure above 0.9
assert acc >= 0.9, f"Accuracy {acc} is below 0.9"

print(f"Test passed! Accuracy = {acc:.2%} (>= 90%)")