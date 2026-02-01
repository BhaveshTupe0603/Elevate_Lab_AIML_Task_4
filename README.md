# 📊 Task 4: Feature Encoding & Scaling

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Pandas](https://img.shields.io/badge/Library-Pandas-150458?style=flat&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange?style=flat&logo=scikit-learn)

## 📖 Overview
**Task 4** focuses on **Feature Engineering**, a critical step in the machine learning pipeline. The objective is to transform raw categorical and numerical data into a format that ML algorithms can process efficiently.

We used the **Adult Income Dataset** to demonstrate techniques such as One-Hot Encoding, Label Encoding, and Standard Scaling.

---

## 🛠️ Tools & Libraries
* **Python:** Core programming language.
* **Pandas:** Data manipulation and analysis.
* **Seaborn/Matplotlib:** Visualization of data distributions.
* **Scikit-Learn:** Used for `StandardScaler`, `LabelEncoder`, and data preprocessing.

---

## ⚙️ Process & Methodology

### 1. Data Cleaning
* Handled missing values (e.g., `?` in the dataset) to ensure data integrity.

### 2. Feature Encoding
Machine learning models cannot process text. We converted categorical data into numbers:
* **Label Encoding:** Applied to the target variable `income` (`<=50K` → `0`, `>50K` → `1`).
* **One-Hot Encoding:** Applied to nominal features like `Race`, `Sex`, and `Workclass` where no inherent order exists. This prevents the model from assuming false rankings.

### 3. Feature Scaling
We applied **StandardScaler** to numerical features (`Age`, `Capital-Gain`, etc.).
* **Goal:** To center the data around a mean of 0 with a standard deviation of 1.
* **Why?** This ensures that features with large values (like `Capital-Gain`) do not dominate the model simply because of their magnitude.

---

## 📉 Results: Before vs. After

### Before Scaling
* **Age:** Range 17 - 90
* **Capital Gain:** Range 0 - 99,999
* *Issue:* Algorithms based on distance (KNN, SVM) would be biased towards Capital Gain.

### After Scaling
* **Age:** Mean ~0.0, Std ~1.0
* **Capital Gain:** Mean ~0.0, Std ~1.0
* *Result:* All features now contribute equally to the model's learning process.

---

## 💻 Code Snippet
Here is how we applied the scaling using Scikit-Learn:

```python
from sklearn.preprocessing import StandardScaler

# Initialize Scaler
scaler = StandardScaler()

# Select numerical features
features_to_scale = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

# Apply transformation
df_encoded[features_to_scale] = scaler.fit_transform(df_encoded[features_to_scale])

# Check new statistics
print(df_encoded[features_to_scale].describe().loc[['mean', 'std']])
```

---
