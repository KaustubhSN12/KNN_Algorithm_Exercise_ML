# 🌸 K-Nearest Neighbors (KNN) Classifier – Iris Dataset

This repository demonstrates an implementation of the **K-Nearest Neighbors (KNN)** algorithm using the classic **Iris dataset**.  
The project covers data preprocessing, training, evaluation, and hyperparameter tuning for K values.

---

## 📌 Overview
- **Algorithm**: K-Nearest Neighbors (Supervised Learning, Classification)  
- **Dataset**: [Iris Dataset – UCI Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)  
- **Goal**: Classify iris flowers into one of three species:
  - *Iris-setosa*  
  - *Iris-versicolor*  
  - *Iris-virginica*  

---

## 📊 Dataset Information
The dataset contains **150 records** with the following features:  
- `sepal-length`  
- `sepal-width`  
- `petal-length`  
- `petal-width`  
- `Class` (Target – Species)  

Example records:  

| sepal-length | sepal-width | petal-length | petal-width | Class           |
|--------------|-------------|--------------|-------------|-----------------|
| 5.1          | 3.5         | 1.4          | 0.2         | Iris-setosa     |
| 7.0          | 3.2         | 4.7          | 1.4         | Iris-versicolor |
| 6.3          | 3.3         | 6.0          | 2.5         | Iris-virginica  |

---

## ⚙️ Steps Covered

### 🔹 1. Data Loading & Exploration
```python
import pandas as pd

names = ['sepal-length','sepal-width','petal-length','petal-width','Class']
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=names)
print(dataset.head())
```
- Checked dataset size, missing values, and statistical description.  

---

### 🔹 2. Train-Test Split
```python
from sklearn.model_selection import train_test_split

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
```

---

### 🔹 3. Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 🔹 4. Model Training & Prediction
```python
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
```

---

### 🔹 5. Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
📌 Example Results:  
- **Confusion Matrix**:  
  ```
  [[10  0  0]
   [ 0 10  1]
   [ 0  0  9]]
  ```  
- **Accuracy**: ~97%  
- **F1-Score (macro avg)**: 0.97  

---

### 🔹 6. Hyperparameter Tuning (Choosing Best K)
```python
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
```

---

## 🔑 Key Learning Outcomes
- Preprocessing and **scaling numerical features**.  
- Implementing **KNN classifier** using Scikit-Learn.  
- Evaluating model with **confusion matrix, accuracy, precision, recall, F1-score**.  
- Using **hyperparameter tuning** to select the optimal K.  

---

## 📚 References
- [Scikit-learn KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)  
- [UCI Iris Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)  

---

## 🔗 Explore My Other Repositories
- 🤖 [Naive Bayes Classifier – Fish Dataset](https://github.com/KaustubhSN12/Naive-Bayes-Fish)  
- 🚀 [K-Means Clustering – Titanic Dataset](https://github.com/KaustubhSN12/KMeans-Clustering-Titanic)  
- 📊 [R Programming Practicals](https://github.com/KaustubhSN12/R-Practice)  
- 🐍 [Python Practice Hub](https://github.com/KaustubhSN12/Python-Practice-Hub)  

---

## 📜 License
This project is licensed under the **MIT License** – free to use and share with credit.  

---

✨ *Star this repository if you found it useful for learning KNN!*  
