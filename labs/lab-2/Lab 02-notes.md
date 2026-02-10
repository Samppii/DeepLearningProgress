# Lab 2: TensorFlow Introduction

> **Goal:** Build your first neural networks using TensorFlow/Keras 
> **Official Docs:** [TensorFlow Documentation](https://www.tensorflow.org/api_docs)

---

## Step 1: What is TensorFlow & Keras?

### TensorFlow
- Google's open-source deep learning library
- Handles the math behind neural networks
- Can run on CPU or GPU

### Keras
- High-level API that runs on top of TensorFlow
- Makes building neural networks simple
- Now integrated into TensorFlow as `tf.keras`

```python
import tensorflow as tf
print(tf.__version__) # Check your version

from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense, Activation
```

### Documentation
- [TensorFlow Guide](https://www.tensorflow.org/guide) 
- [Keras Documentation](https://keras.io/)

---

### Step 2: Classification vs Regression

Neural networks can do two main types of predictions:

### Regression
- **Output:** A continuous number
- **Example:** Predict car's MPG (18.5, 24.3, 31.0)
- **Output Layer:** Single Neuron
- **Loss Function:** Mean Squared Error

```python
model.add(Dense(1)) # One output neuron
model.compile(loss='mean_squared_error', optimizer='adam')
```

### Classification
- **Output:** A category/class
- **Example:** Predict flower species (setosa, versicolor, virginica)
- **Output Layer:** One neuron per class
- **Loss Function**: Categorical Cross-Entropy
```python
model.add(Dense(3, activation='softmax')) # 3 classes = 3 neurons
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

### Quick Comparison

|                   | Regression         | Classification           |
| ----------------- | ------------------ | ------------------------ |
| Predicting        | Number             | Category                 |
| Output Neurons    | 1                  | # of classes             |
| Output Activation | None (Linear)      | Softmax                  |
| Loss Function     | mean_squared_error | categorical_crossentropy |
| Metric            | RMSE               | Accuracy                 |

---

## Step 3: Neural Network Architecture

### The Three Layer Types

```
Input Layer → Hidden Layer(s) → Output Layer
```

**Input Layer:**
- Receives your feature data
- Size = number of features in your data
- Has a bias neuron

**Hidden Layer(s):**
- Where the "learning" happens
- You choose how many layers and neurons
- Has bias neurons
- Uses activation functions (usually ReLU)

**Output Layer:**
- Produces the final prediction
- No bias neuron
- Size depends on task (1 for regression, # classes for classification)

### Building a Model in Keras

```python
from tensorflw.keras.model import Sequental
from tensorflow.keras.layers import Dense

model = Sequential()

# Hidden Layer 1: 50 neurons, ReLU activation
# input_dim = number of features in your data
model.add(Dense(50, input_dim = x.shape[1], activation='relu')) # x.shape = (398, 9) # x.shape[0] = 398 - rows (samples) # x.shape[1] - columns (features)

# Hidden layer 2: 25 neurons, ReLU activation
model.add(Dense(25, activation = 'relu'))

# Output layer: 1 neuron (for regression)
model.add(Dense(1))
```

### What is `Dense`?
A "Dense" layer means every neuron connects to every neuron in the previous layer. Also called a "fully connected" layer.

### What is `activation`?
The activation function decides whether a neuron "fires" or not.

| Activation  | Use Case                        |
| ----------- | ------------------------------- |
| `relu`      | Hidden layers (most common)     |
| `softmax`   | Output layer for classification |
| None/Linear | Output layer for regression     |

### Documentation:
- [Sequential Model](https://www.tensorflow.org/guide/keras/sequential_model)
- [Dense Layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)

---

## Step 4: Data Preprocessing Helper Functions

Before feeding data to a neural network, you need to preprocess it. Helper functions from the lab:

### 4.1 Handle Missing Values

```python
def missing_median(df, name):
	"""Fill missing values with the column median"""
	med = df[name].median() # get middle value of that column
	df[name] = df[name].fillna(med) # replace the NAN with the median value
	
# Usage
missing_median(df, 'horsepower')
```

### 4.2 Encode Categorical Data (One-Hot / Dummy)

For **input features** (predictors):

```python
def encode_text_dummy(df, name):
"""Convert categories to dummy variables red, green, blue -> [1,0,0], [0,1,0], [0,0,1]"""

dummies = pd.get_dummies(df[name])
for x in dummies.columns:
	dummy_name = "{}-{}".format(name, x)
	df[dummy_name] = dummies[x]
	df.drop(name, axis=1, inplace=True)
	
# Usage
encode_text_dummy(df, 'origin') # Creates origin-1, origin-2, origin-3
```

### 4.3 Encode Categorical Data (Label Encoding)

For **output/target** variable:

```python
def encode_text_index(df, name):
	"""Convert categories to integers setosa, vesicolor, virginica -> 0, 1, 2"""
	le = preprocessing.LabelEncoder()
	df[name] = le.fit_transform(df[name])
	return le.classes_ # Returns the original class names
	
# Usage
species = encode_text_index(df, 'species')
# Species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginca']
```

### 4.4 Normalize Numeric Data (Z-Score)

```python
def encode_numeric_zscore(df, name, mean=None, sd=None):
	"""Normalize using z-score: (value - mean) / std"""
	if mean is None:
		mean = df[name].mean()
	if sd is None:
		sd = df[name].std()
	df[name] = (df[name] - mean) / sd
	
# Usage
encode_numeric_zscore(df, 'horsepower')
encode_numeric_zscore(df, 'weight')
```

**Why normalize?** Neural Networks train faster and better when all features are on similar scales.

### 4.5 Convert DataFrame to X, y

```python
def to_xy(df, target):
	"""
	Split dataframe into:
	- x: feature matrix (all columns expect target)
	- y: target vector (the column you're predicting)
	  
	Automatically handles classification vs regression
	"""
	result = []
	for x in df.columns:
		if x != target:
			result.append(x)
			
	target_type = df[target].dtypes
	
	if target_type in (np.int64, np.int32):
		# Classification - one-hot encode target
		dummies = pd.get_dummies(df[target])
		return df[result].values.astype(np.float32), dummies.values.astype(np.float32)
	else:
		# Regression - keep target as-is
		return df[result].values.astype(np.float32), df[target].values.astype(np.float32)
		
# Usage
x, y = to_xy(df, 'mpg') # Regression
x, y = to_xy(df, 'species') # Classification
```

### Preprocessing Workflow Summary

**For Input Features (x):**
1. Fill missing values → `missing_median()`
2. Encode text/categorical → `encode_text_dummy()`
3. Normalize numeric → `encode_numeric_zscore()`

**For Output/Target (y):**
1. Drop rows with missing target
2. Encode text/categorical → `encode_text_index()`
3. Do NOT normalize numeric targets

**Finally:**
- Convert to x, y → `to_xy()`

---

## Step 5: Regression Example - MPG prediction

Predict a car's fuel efficiency (MPG) based on its features/

### 5.1 Load and Explore Data

```python
import pandas as pd
import numpy as np
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

path = "./data/"
df = pd.read_csv(path + "auto-mpg.csv", na_values=['NA', '?'])
df.head()
```

### 5.2 Preprocess Data

```python
# Save car names for later (not a feature)
cars = df['name']
df.drop('name', axis=1, inplace=True)

# Fill missing horsepower with median
missing_median(df, 'horsepower')

# Encode categorical 'origin' as dummy variables
encode_text_dummy(df, 'orgin')

# Convert to x, y
x, y = to_xy(df, 'mpg')

print(x.shape) # (398, 9) - 398 cars, 9 features
print(y.shape) # (398, ) - 398 MPG values
```

### 5.3 Build the Model

```python
model = Sequential()

# Hidden layer 1: 50 neurons
model.add(Dense(50, input_dim=x.shape[1], activation='relu'))

# Hidden layer 2: 25 neurons
model.add(Dense(25, activation='relu'))

# Output layer: 1 neuron (regression)
model.add(Dense(1))

# Compile
model.compile(loss='mean_squared_error', optimizer='adam')
```

