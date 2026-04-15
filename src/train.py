# imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# create ouput directory
os.makedirs("outputs", exist_ok=True)

# 1. data
iris = load_iris()
X = iris.data
y = iris.target

# 2. 80/20 split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 4. predictions
predictions = model.predict(X_test)

# 5. acc calc
accuracy = accuracy_score(y_test, predictions)
print(f"The AI is {accuracy * 100:.2f}% accurate")

# 6. confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# 7. confusion matrix to png
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Iris Classifier')
plt.savefig('outputs/confusion_matrix.png')
plt.close()

print("\nConfusion matrix saved to outputs/confusion_matrix.png")