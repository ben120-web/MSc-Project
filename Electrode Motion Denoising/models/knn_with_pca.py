from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

pca = PCA(n_components=10)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)  # Adjust the number of neighbors as needed
knn.fit(X_train_pca, y_train)

accuracy = knn.score(X_test_pca, y_test)
print(f"PCA with KNN Accuracy: {accuracy}")
