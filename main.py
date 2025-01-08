import pandas as pd
df = pd.read_csv('parkinsons.csv')
df.head()

features = ["PPE", "DFA"]
target = "status"
X = df[features]
y = df[target]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

import joblib
joblib.dump(knn, "my_model.joblib")
print("Model saved successfully!")
