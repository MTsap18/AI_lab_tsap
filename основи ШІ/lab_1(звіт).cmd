Лабораторна робота №1
Тема: Підключення набору даних, обробка даних та побудова кореляційної матриці
Мета роботи: Формування навиків та умінь при роботі з даними.

1. Завантаження та підключення датасету:
   Для початку необхідно підключити необхідні бібліотеки, завантажити набір даних та перевірити його на наявність пропущених значень. 

   Виконано наступне:
   - Імпортовані бібліотеки:
     - numpy для роботи з масивами та математичними операціями.
     - pandas для роботи з табличними даними.
     - seaborn для візуалізації.
     - matplotlib для побудови графіків.
   
   - Завантажено дані з файлу «advertising.csv»:
     ```python
     advertising = pd.read_csv('advertising.csv')
     ```

2. Обробка даних:
   Для забезпечення коректності аналізу були видалені всі рядки з пропущеними значеннями (NaN). Це забезпечує точність побудови кореляційної матриці:
   ```python
   advertising.dropna(inplace=True)

3. Побудова кореляційної матриці: Для оцінки взаємозв'язків між різними змінними був обчислений коефіцієнт кореляції між ними:

    advertising_correlation = advertising.corr()

4. Візуалізація кореляційної матриці: Для зручності аналізу кореляцій була побудована теплова карта, яка відображає рівень кореляції між різними змінними:
    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(advertising_correlation, annot=True, cmap='coolwarm', fmt=".2f")
    heatmap.set_title('Матриця кореляції ознак Advertising', fontsize=16)
    plt.show()

    Матриця: "лабораторна_1.jpg".
5. Висновки:
    Після обробки даних та побудови кореляційної матриці ми можемо побачити, які змінні мають сильний або слабкий взаємозв'язок між собою.
    Теплова карта допомагає візуалізувати ці взаємозв'язки, що дозволяє визначити важливі фактори для подальшого аналізу чи моделювання.