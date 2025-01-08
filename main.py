import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib 

# קריאת הנתונים
df = pd.read_csv("parkinsons.csv")
df = df.dropna()

# בחירת מאפיינים ומטרה
features = ["PPE", "DFA"]
target = "status"
X = df[features]
y = df[target]

# נרמול הנתונים
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# פיצול הנתונים
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# אימון המודל
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# בדיקת המודל
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# שמירת המודל
joblib.dump(knn, "Ronny_model.joblib")
print("Model saved successfully!")
