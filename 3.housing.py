import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('./data/3.housing.csv', index_col=0, delim_whitespace=True, names = header)

array = data.values
# 독립변수 / 종속변수
X = array[:, 0:13]
Y = array[:, 13]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)

model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

plt.scatter(X_test[:15], Y_test[:15], color = 'blue')
plt.scatter(X_test[:100], Y_test[:100], color = 'red', marker= '*')
plt.xlabel('Index')
plt.ylabel('Medv($1,000)')
plt.show()
mes = mean_squared_error(Y_test, y_pred)
print(mes)

kfold = KFold(n_splits=5)
mes = cross_val_score(model, X, Y, scoring = " ")
print()
