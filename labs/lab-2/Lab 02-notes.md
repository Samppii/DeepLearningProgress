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

For **input features** (predictors)L

```python
def encode_text_dummy(df, name):
"""Convert categories to dummy variables red, green, blue -> [1,0,0], [0,1,0], [0,0,1]"""

dummies = pd.get_dummies(df[name])
for x in dummies.columns:
	dummy_name = "{}-{}".format(name, x)
	df[dummy_name] = dummies[x]
	df.drop(name, axis=1, inplace=True)
	
# Usage
encode_text_dummy(df, 'origin) # Creates origin-1, origin-2, origin-3
```
