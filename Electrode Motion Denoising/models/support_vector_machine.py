from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming X is your feature set and y is your labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"SVM Accuracy: {accuracy}")
