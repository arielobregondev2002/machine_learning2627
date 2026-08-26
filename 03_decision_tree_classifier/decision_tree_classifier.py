from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.DataFrame({
 "horas_estudio": [1, 2, 2, 3, 4, 4, 5, 6, 7, 8],
 "horas_sueño": [8, 7, 6, 8, 7, 5, 8, 6, 7, 8],
 "aprobado": [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
})

X = df[['horas_estudio', 'horas_sueño']]
y = df['aprobado']

#Create the decision tree model
model = DecisionTreeClassifier(criterion='gini', random_state= 42)

#Divide features and targets into the ones I'll use for training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state=42)

#Training the model
model.fit(X_train, y_train)

#Prediction using features for testing
prediction = model.predict(X_test)

print(X_test)
print(y_test)
print(prediction)

#Accuracy of prediction
accuracy = accuracy_score(y_test, prediction)
print("accuracy:", accuracy)

#Show the decision tree
plt.figure()
plot_tree(model)
plt.title("Decicion Tree")
plt.show()