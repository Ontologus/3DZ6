import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
X = 2 * np.random.rand(100, 1)
y = 10 + 5 * X + np.random.randn(100, 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
plt.scatter(X_train, y_train, c='blue', label='train')
plt.scatter(X_test, y_test, c='green', label='test')
plt.plot(X_test, y_pred, c='black', label='linear regression')
plt.grid()
plt.legend()
plt.show()