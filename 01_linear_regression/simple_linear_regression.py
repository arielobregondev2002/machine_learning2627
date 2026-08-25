import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

db = pd.DataFrame({
    "Horas estudiadas": [1, 2, 3, 4, 5, 6, 7, 8],
    "Nota": [52, 58, 65, 71, 78, 85, 91, 96]
})

X = db[['Horas estudiadas']].values
y = db['Nota'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(X_train, y_train)

prediction= model.predict(X_test)
print(X_test)
print(y_test)
print(prediction)

mse = mean_squared_error(y_test, prediction)
print("Error:", mse)
print("coef:", model.coef_)
print("interception:", model.intercept_)
