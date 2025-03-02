import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Завантажуємо дані
data = pd.read_csv('advertising.csv')

# Перевіримо перші кілька рядків даних
print(data.head())

# Припустимо, що дані мають стовпці "TV", "Radio", "Newspaper" та "Sales"
# Використовуємо всі стовпці, окрім "Sales" для кластеризації
X = data[['TV', 'Radio', 'Newspaper']]

# Стандартизуємо дані
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Створюємо модель KMeans з 3 кластерами
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Додаємо мітки кластерів до даних
data['Cluster'] = kmeans.labels_

# Виводимо кількість зразків в кожному кластері
print(data['Cluster'].value_counts())

# Візуалізація результатів
plt.figure(figsize=(10, 6))
plt.scatter(data['TV'], data['Radio'], c=data['Cluster'], cmap='viridis')
plt.xlabel('TV')
plt.ylabel('Radio')
plt.title('K-Means Clustering of Advertising Data')
plt.colorbar(label='Cluster')
plt.show()
