import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump

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
model = SVC()
model.fit(X_train, y_train)

# בדיקת המודל
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# שמירת המודל
dump(model, "Ronny_model.joblib")
print("Model saved successfully!")
