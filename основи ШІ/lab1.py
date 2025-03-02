import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Завантаження датасету
advertising = pd.read_csv('advertising.csv')

# Видалення рядків з пропущеними значеннями
advertising.dropna(inplace=True)

# Побудова кореляційної матриці
advertising_correlation = advertising.corr()

# Візуалізація кореляційної матриці
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(advertising_correlation, annot=True, cmap='coolwarm', fmt=".2f")
heatmap.set_title('Матриця кореляції ознак Advertising', fontsize=16)
plt.show()

