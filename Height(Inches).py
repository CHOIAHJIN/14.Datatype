import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv('./data/5.HeightWeight.csv', index_col=0)
data['Height(Inches)'] = data['Height(Inches)']*2.54
data['Weight(Pounds)'] = data['Weight(Pounds)']*0.453592
array = data.values
X = array[:,0]
print(X)
Y = array[:,1]
X= X.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, Y_train)
y_prediction = model.predict(X_test)
mae = mean_absolute_error(y_prediction, Y_test)
print(mae)

plt.scatter(X_test[:100], y_prediction[:100], color = 'red')
plt.scatter(X_test[:100], Y_test[:100], color = 'blue')
plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')
plt.show()