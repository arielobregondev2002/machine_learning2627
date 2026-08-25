import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
db = pd.DataFrame({
    "Horas estudiadas": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8],
    "Aprobó": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
})

X = db[['Horas estudiadas']]
y = db['Aprobó']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

model = LogisticRegression()

model.fit(X_train, y_train)
probability = model.predict_proba(X_test)
prediction = model.predict(X_test)

print(X_test)
print(y_test)
print(probability)
print(prediction)

accuracy = accuracy_score(y_test, prediction)
print(accuracy