# Lecture 2: Machine Learning Principles
## CSC 296S - Deep Learning | Dr. Victor Chen

---

## 1. What is Machine Learning?

**Definition:** A study on getting a computer to finish a task *without explicitly programming it.*

Instead of writing rules by hand (e.g., "If pixel patterns look like X, classify as cat"), you let the algorithm **learn the patterns from data.**

---

## 2. The Two Major Learning Types

### 2.1 Supervised Learning
- **All the data are labeled** - you have input-output pairs
- The model learns the mapping - input → correct output
- **This is the main focus of early deep learning**

### 2.2 Unsupervised Learning
- **Data has no labels** - you only have inputs
- The model finds hidden structure/patters in the data

---

## 3. Machine Learning in a Nutshell (The 2x2 Matrix)

Crucial Table - memorize it:

|                       | **Supervised Learning**                                                               | **Unsupervised Learning**                       |
| --------------------- | ------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Continuous Output** | **Regression** (Linear regression, Neural Networks)                                   | Dimensionality Reduction (PCS, t-SNE)           |
| **Discrete Output**   | **Classification** (KNN, SVM, decision tree, Bayes, logistic regression, neural nets) | Clustering (K-means, hierarchical, DBSCAN, GMM) |

**Key Insight:**
- If you're predicting a *number* (price, temperature) → Regression
- If you're predicting a *category* (spam/not spam, cat/dog) → Classification
- If you're grouping without labels → Clustering
- If you're compressing features → Dimensionality Reduction

---

## 4. Supervised Learning Deep Dive

### 4.1 The Pipeline

``` 
Training Phase: 
┌───────────────┐   ┌──────────┐   ┌─────────┐
│ Training Data │ → │ Features │ → │ Model   │ → Learned Parameters (Θ)
│ + Labels      │   │          │   │ Training│
└───────────────┘   └──────────┘   └─────────┘
Inference Phase: 
┌─────────────────┐   ┌──────────┐   ┌───────┐
│ New Test Sample │ → │ Features │ → │ Model │ → Predicted Label
└─────────────────┘   └──────────┘   │ y=f(x)│
                                     └───────┘
```


### 4.2 Steps in Supervised Learning

1. **Define the problem** - What are you trying to predict?
2. **Identify input features** - What helps discriminate between classes?
3. **Identify output labels** - What categories/values are you predicting?
4. **Decide on a model** - What's the right model for your problem?
5. **Train the model** on training data
6. **Test the model** on test data (data it has never seen)

### 4.3 The Parametric Approach

**Core equation:**
```
y = f_Θ(x)
```

Where: 
- `x` = input (features)
- `y` = output (label/prediction)
- `Θ` = parameters/weights the model learns 
- `f` = the function the model represents

**Example - Image Classification:**
```
f(x, W): [32x32x3 image] → [10 class scores]
		 (3072 numbers)
```

The model learns weight `W` that map pixel values to class probabilities.

---

## 5. Linear Regression (Simplest Parametric Model)

### 5.1 Single Variable

```
y = α + β * x
```
 - `α`  = y-intercept (where the line crosses y-axis)
 - `β` = slope (rise over run, Δy/Δx)

### 5.2 Multiple Variables 

```
y = α + β₁*X₁ + β₂*X₂ + ... + βₖ*Xₖ
```

- `k` = number of input variables/features

### 5.3 How Do We Measure Error?

**Sum of Squared Errors (SSE):**
```
SSE = Σ(actual - predicted)²
``` 

The model's goal is to **minimize SSE** - find the line that has the smallest total squared distance from all points.

---
## 6. Unsupervised Learning Deep Dive

### 6.1 Clustering

**Goal:** Divide "similar" data points into the same groups - without any labels.

**K-Means Clustering:**
1. Pick K cluster centers (randomly or strategically)
2. Assign each point to its near center
3. Recalculate centers as the mean of assigned points
4. Repeat until convergence

**Objective:** Minimize Sum of Squared Error (SSE) within clusters
**Visualization:** https://www.naftaliharris.com/blog/visualizing-k-means-clustering/

### 6.2 Dimensionality Reduction

**Why reduce dimensions?**
- Curse of dimensionality - models struggle with too many features
- Visualization - can't visualize 100D data, but can visualize 2D/3D
- Noise reduction - removing redundant features
- Computational efficiency - fewer features = faster training

**Principal Component Analysis (PCA):**
- Finds new orthogonal basis vectors that capture maximum variance
- Projects high-dimensional data onto lower dimensions while preserving as much information as possible

**Example:**
- 3D points can sometimes be represented in 2D if they lie on a plane
- PCA finds that plane automatically

**t-SNE:** Another popular technique for visualization (especially for high-dimensional data like images)

---

## 7. K-Nearest Neighbors (KNN) Classifier

**How it works:**
1. Store all training examples
2. For a new test point, find the **K closest** training points
3. **Majority Vote** - assign the most common label among those k neighbors

**Key Properties:**
- No explicit training phase (lazy learner)
- Simple but can be slow of large datasets
- Choice of K matters (too small = noisy, too large = blurred boundaries)

---

## 8. Scikit-Learn (sklearn)

**The go-to Python Library for classical ML.**

All supervised models in sklearn share the **same interface:**
```python
from sklearn.model_family import ModelName

# Create Model
mode = ModelName()

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
score = model.score(X_test, y_test)
```

This consistency makes it easy to swap models and experiment.

---

## Data Preprocessing

### 9.1 Understanding Data

**Objects (rows):** Entities in the world (person, transaction, image)
- Also called: record, point, sample, instance

**Attributes (columns):** Properties of an object (name, age, height)
- Also called: field, feature

### 9.2 Attribute Types

| Type            | Subtypes                              | Examples                                   |
| --------------- | ------------------------------------- | ------------------------------------------ |
| **Numeric**     | Discrete, Continuous                  | dates, temperature, time, length           |
| **Categorical** | Nominal (no order), Ordinal (ordered) | eye color, rankings, {tall, medium, short} |

---

## 10. Data Encoding (Categorical → Numeric)

**Why?** ML models work with numbers. We need to convert categories to numeric form.

### 10.1 Label Encoding
Assign one unique number to each distinct value:

```
Single → 0
Married → 1
Divorced → 2
```

⚠️ **NEVER use label encoding on input features!**

Why? It implies an ordering that doesn't exist. The model might think Divorced (2) > Married (1) > Single (0), which is meaningless.

### 10.2 One-Hot Encoding (The Right Way)

Create a **new binary column** for each distinct value:

**Before:**

| Martial Status |
| -------------- |
| Single         |
| Married        |
| Divorced       |

**After:**

| **Single** | **Married** | **Divorced** |
| ---------- | ----------- | ------------ |
| 1          | 0           | 0            |
| 0          | 1           | 0            |
| 0          | 0           | 1            |
**Key Insight:** This turns records into vectors, enabling linear algebra operations.

---

## 11. Data Normalization (Numeric Features)

**Why?** Different features have different scales. Without normalization:
- Temperature: 24-32
- Humidity: 0.3-0.8
- Pressure: 80-95

A model might think Pressure is more "important" just because it has bigger numbers.

### 11.1 Option 1: Max Normalization

```
new_value = old_value / max_value
```

Range: [ 0 , 1 ] (assuming positive values)

### 11.2 Option 2: Min-Max Normalization

```
new_value = (old_value - min) / (max - min)
```

Range: [ 0 , 1 ]

### 11.3 Option 3: Z-Score Standardization (Most Common)

```
z = (x - μ) / σ
```

Where:
- μ = mean of the column
- σ = standard deviation

**Properties:**
- Mean becomes 0, standard deviation becomes 1
- Negative z-score = below average
- Positive z-score = above average

---

## 12. Model Evaluation

### 12.1 Train/Test Split

**Golden Rule:** Never evaluate on data you trained on!

```
All Data
├── Training Set (~70%) → Used to learn parameters 
└── Test Set (~30%) → Used to evaluate performance
```

### 12.2 The Ultimate Goal

> **MAXIMIZE the ability of the model to predict data that it has NOT already seen.**

This is called **generalization**.

---

## 13. Underfitting vs Overfitting

This is one of the most important concepts in all of ML:

| Problem          | What's Happening  | Training Error | Test Error |
| ---------------- | ----------------- | -------------- | ---------- |
| **Underfitting** | Model too simple  | High           | High       |
| **Good Fit**     | Model just right  | Low            | Low        |
| **Overfitting**  | Model too complex | Very Low       | High       |

**Underfitting:** The model can't even capture the training data patterns.
**Overfitting**: The model memorizes the training data (including noise) but fails to generalize to new data.

### 13.1 How to Avoid?

- **Underfitting:** Use a more complex model, add features, train longer
- **Overfitting:** Use a simpler mode, get more data, regularization, dropout (in neural nets)

---

## 14. Evaluation Metrics

### 14.1 Classification Metrics

#### Confusion Matrix (Binary Classification)

|                 | **Predicted: No**   | **Predicted: Yes**  |
| --------------- | ------------------- | ------------------- |
| **Actual: No**  | True Negative (TN)  | False Positive (FP) |
| **Actual: Yes** | False Negative (FN) | True Positive (TP)  |
#### Key Metrics

**Precision:** Of all predicted positives, how many were actually positive?

```
Precision = TP / (TP + FP)
```

*"When I say yes, am I right?"*

**Recall (Sensitivity): Of all actual positives, how many did we catch?**

```
Recall = TP / (TP + FN)
```

*"Did I find all the positives?"*

**F1-Score:** Harmonic mean of precision and recall

```
F1 = 2 x (Precision x Recall) / (Precision + Recall)
```

*Single number combining both - useful when you need one metric*

#### ROC Curve & AUROC

**ROC Curve:** Plots True Positive Rate (TPR) vs False Positive Rate (FPR) at various thresholds

**AUROC (Area Under ROC Curve):**
- Perfect model: AUROC = 1.0
- Random guessing: AUROC = 0.5
- Higher is better

#### Cross-Entropy (Log Loss)

Measures how close predicted probability distribution is to true distribution.

```
Cross-Entropy = -Σ y_true × log(y_pred)
```

- Perfect prediction: Cross-entropy = 0
- Higher cross-entropy = worse predictions
- **This is the loss function used in most classification neural networks**

### 14.2 Regression Metrics

**MAE (Mean Absolute Error):**

```
MAE = (1/n) × Σ|actual - predicted|
```
*Average magnitude of errors*

**MSE (Mean Squared Error):**

 ``` 
 MSE = (1/n) × Σ(actual - predicted)² 
 ```
 *Penalizes large errors more heavily* 

**RMSE (Root Mean Squared Error):**

``` 
RMSE = √MSE
``` 
*Same units as the target variable — more interpretable*

---

## 15. Key Takeaways from Lecture 2

1. **ML = Learning from data** instead of explicit programming.
2. **Supervised vs Unsupervised** - Labeled data vs finding hidden structure.
3. **The 2x2 matrix:** Classification/Regression x Supervised/Unsupervised covers most ML tasks.
4. **Parametric models** learn parameters 0 such that y = f_Θ(x).
5. **Data preprocessing is critical:**
	- Categorical features → One-hot encoding (NOT label encoding for inputs!) 
	- Numeric features → Normalization (z-score is most common)
6. **Train/test split** - always evaluate on unseen data.
7. **Underfitting** = model too simple, **Overfitting** = model too complex. Find the sweet spot.
8. **Know your metrics:**
	- Classification: Precision, Recall, F1, AUROC, Cross-entropy
	- Regression: MAE, MSE, RMSE
9.  **sklearn** provides a consistent interface for all classical ML models.
10. **Cross-entropy** will become extremely important - it's the loss function for neural network classifiers.

---

## Quick Reference: sklearn Pattern

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use same scaler

# Train
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))
```

