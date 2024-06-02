from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

accuracy = rf.score(X_test, y_test)
print(f"Random Forest Accuracy: {accuracy}")
