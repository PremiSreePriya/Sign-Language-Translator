# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("dataset.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
