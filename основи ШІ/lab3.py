import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Завантажуємо дані
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing = pd.read_csv('housing.csv', header=None, delimiter=r"\s+", names=column_names)

# Визначаємо вхідні ознаки та цільову змінну
X = housing.drop('MEDV', axis=1)  # Усі стовпці, крім 'MEDV' (середня вартість нерухомості)
y = housing['MEDV']  # Цільова змінна - MEDV (середня вартість нерухомості)

# Розділяємо на навчальні та тестові дані
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ініціалізація та навчання моделі дерева рішень
decision_tree = DecisionTreeRegressor(random_state=42)
decision_tree.fit(X_train, y_train)

# Прогнозування на тестових даних
y_pred = decision_tree.predict(X_test)

# Обчислюємо середньоквадратичну помилку (MSE) та коефіцієнт детермінації (R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Виводимо результат
print(f"Середньоквадратична помилка (MSE): {mse}")
print(f"Коефіцієнт детермінації (R²): {r2}")

# Побудова графіка дерева рішень
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, feature_names=X.columns, filled=True, rounded=True)
plt.show()
