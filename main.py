import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# 1. קריאת הנתונים וניקוי
df = pd.read_csv("parkinsons.csv")
df = df.dropna()

# 2. הגדרת המאפיינים והיעד
features = ['PPE', 'DFA']  # המאפיינים המקוריים
target = "status"
X = df[features]
y = df[target]

# 3. נירמול הנתונים
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. פיצול הנתונים לסט אימון וסט בדיקה
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. אימון המודל
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)  # כוונון פרמטרים
model.fit(X_train, y_train)

# 6. בדיקת דיוק המודל
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

# 7. שמירת המודל
dump(model, "model.joblib")
