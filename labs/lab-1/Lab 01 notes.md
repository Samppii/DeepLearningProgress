# Lab 1: Data Preprocessing with Pandas

> CSC 296S Deep Learning  
> **Official Docs:** [Pandas Documentation](https://pandas.pydata.org/docs/)

---
## Step 1: What is Pandas & Why Do We Need It?

### The Big Picture

NumPy = great for numerical arrays Pandas = great for **tabular data** (rows and columns, like spreadsheets/CSVs)

```python
import pandas as pd   # standard alias
import numpy as np
```

### Two Main Data Structures

|Structure|What It Is|Like...|
|---|---|---|
|**Series**|1D labeled array|A single column|
|**DataFrame**|2D labeled table|A spreadsheet/CSV|

```python
# Series - one column
s = pd.Series([1, 2, 3, 4])

# DataFrame - multiple columns
df = pd.DataFrame({
    'name': ['Alice', 'Bob'],
    'age': [25, 30]
})
```

### Why Pandas for Deep Learning?

1. Load data from CSV, Excel, databases
2. Clean messy data (missing values, wrong types)
3. Transform data (normalize, encode categories)
4. Prepare feature vectors for TensorFlow/PyTorch

### Documentation:

- [10 Minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)
- [Pandas Overview](https://pandas.pydata.org/docs/getting_started/overview.html)

---

## Step 2: Reading Data Files

### 2.1 Reading CSV Files

```python
import pandas as pd

# Basic CSV read
df = pd.read_csv('data/auto-mpg.csv')

# With custom options
df = pd.read_csv('data/auto-mpg.csv', na_values=['NA', '?'])
#                                     ↑ treat these as missing values
```

### 2.2 Reading Files with Different Separators

```python
# Tab-separated file (.tsv)
orders = pd.read_csv('data/chipotle.tsv', sep='\t')

# Pipe-separated file
users = pd.read_csv('data/u.user', sep='|', header=None, names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])
#                                           ↑ no header   ↑ provide column names
```

### 2.3 Quick Data Inspection

```python
df.head()        # First 5 rows
df.head(10)      # First 10 rows
df.tail()        # Last 5 rows
df.shape         # (rows, columns)
df.columns       # Column names
df.dtypes        # Data type of each column
df.info()        # Summary info
df.describe()    # Statistics for numeric columns
```

### Documentation:

- [pd.read_csv()](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

### Practice:

```python
# Load these files and inspect them:
# 1. Load 'data/chipotle.tsv' (tab-separated)
# 2. Load 'data/ufo.csv'
# 3. Check how many rows and columns each has
# 4. Look at the first 5 rows of each
```

---

## Step 3: DataFrame Basics

### 3.1 What's in a DataFrame?

```python
df = pd.read_csv('data/auto-mpg.csv')

# See the structure
print(type(df))           # pandas.core.frame.DataFrame
print(df.shape)           # (398, 9) - 398 rows, 9 columns
print(df.columns)         # Index(['mpg', 'cylinders', ...])
```

### 3.2 Data Types

```python
df.dtypes
# mpg             float64
# cylinders         int64
# displacement    float64
# horsepower       object   ← string! might need cleaning
# weight            int64
# ...
```

Common dtypes:

- `int64` - integers
- `float64` - decimals
- `object` - strings (text)
- `bool` - True/False

### 3.3 Basic Statistics

```python
df.describe()
# Shows count, mean, std, min, 25%, 50%, 75%, max for numeric columns
```

### Documentation:

- [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)

---

## Step 4: Selecting Data (Series & Columns)

### 4.1 Selecting a Single Column (Returns Series)

```python
# Two ways - both return a Series
df['mpg']          # Bracket notation (always works)
df.mpg             # Dot notation (only for simple column names)
```

### 4.2 Selecting Multiple Columns (Returns DataFrame)

```python
# Pass a list of column names
df[['mpg', 'cylinders', 'horsepower']]
```

### 4.3 Selecting Rows by Index Position: `iloc`

```python
# iloc = integer location (position-based)
df.iloc[0]           # First row
df.iloc[0:5]         # First 5 rows
df.iloc[-1]          # Last row
df.iloc[0, 2]        # Row 0, Column 2
df.iloc[0:5, 0:3]    # First 5 rows, first 3 columns
```

### 4.4 Selecting Rows by Label: `loc`

```python
# loc = label-based
df.loc[0]                    # Row with index label 0
df.loc[0:5]                  # Rows 0 through 5 (inclusive!)
df.loc[0, 'mpg']             # Row 0, column 'mpg'
df.loc[0:5, 'mpg':'weight']  # Rows 0-5, columns mpg through weight
```

### Key Difference: `iloc` vs `loc`

||iloc|loc|
|---|---|---|
|Uses|Integer position|Label/name|
|Slicing|Exclusive end|Inclusive end|
|Example|`df.iloc[0:3]` → rows 0,1,2|`df.loc[0:3]` → rows 0,1,2,3|

### 4.5 Boolean Filtering

```python
# Filter rows where mpg > 30
df[df['mpg'] > 30]

# Multiple conditions (use & for AND, | for OR)
df[(df['mpg'] > 30) & (df['cylinders'] == 4)]

# Filter by string
df[df['name'].str.contains('ford')]
```

### Documentation:

- [Indexing and Selecting](https://pandas.pydata.org/docs/user_guide/indexing.html)
- [iloc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html)
- [loc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html)

### Practice:

```python
df = pd.read_csv('data/auto-mpg.csv')

# Try:
# 1. Get the 'name' column
# 2. Get columns 'mpg', 'horsepower', 'name'
# 3. Get the first 10 rows
# 4. Get row 5, column 'name' using iloc
# 5. Filter: cars with mpg > 35
# 6. Filter: cars with 4 cylinders AND mpg > 30
```

---

## Step 5: Handling Missing Values

### 5.1 Finding Missing Values

```python
# Check for missing values
df.isnull()              # DataFrame of True/False
df.isnull().sum()        # Count missing per column
df.isnull().sum().sum()  # Total missing values

# Find rows with missing values in a specific column
df[df['horsepower'].isnull()]
```

### 5.2 Dropping Missing Values

```python
# Drop all rows with ANY missing values
df_clean = df.dropna()

# Drop rows where specific column is missing
df_clean = df.dropna(subset=['horsepower'])
```

### 5.3 Filling Missing Values

```python
# Fill with a specific value
df['horsepower'] = df['horsepower'].fillna(0)

# Fill with the median (common for numeric data)
median_hp = df['horsepower'].median()
df['horsepower'] = df['horsepower'].fillna(median_hp)

# Fill with the mean
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())
```

### 5.4 Reading Data and Specifying NA Values

```python
# Tell Pandas what counts as "missing"
df = pd.read_csv('data/auto-mpg.csv', na_values=['NA', '?', 'N/A', ''])
```

### Documentation:

- [Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [fillna()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html)
- [dropna()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)

### Practice:

```python
df = pd.read_csv('data/auto-mpg.csv', na_values=['NA', '?'])

# Try:
# 1. How many missing values in each column?
# 2. Which rows have missing horsepower?
# 3. Fill missing horsepower with the median
# 4. Verify no more missing values in horsepower
```

---

## Step 6: Sorting & Filtering

### 6.1 Sorting

```python
# Sort by one column
df_sorted = df.sort_values(by='mpg')                    # Ascending (default)
df_sorted = df.sort_values(by='mpg', ascending=False)   # Descending

# Sort by multiple columns
df_sorted = df.sort_values(by=['cylinders', 'mpg'], ascending=[True, False])
```

### 6.2 Filtering (Boolean Indexing)

```python
# Single condition
high_mpg = df[df['mpg'] > 30]

# Multiple conditions
efficient_4cyl = df[(df['mpg'] > 30) & (df['cylinders'] == 4)]

# Using isin() for multiple values
japanese = df[df['origin'].isin([2, 3])]

# String filtering
fords = df[df['name'].str.contains('ford', case=False)]
```

### 6.3 Value Counts

```python
# Count unique values in a column
df['cylinders'].value_counts()
# 4    204
# 8    103
# 6     84
# ...
```

### Documentation:

- [sort_values()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html)
- [value_counts()](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html)

---

## Step 7: Modifying DataFrames

### 7.1 Adding New Columns

```python
# Simple calculation
df['weight_kg'] = df['weight'] * 0.453592

# Using insert() to specify position
df.insert(1, 'weight_kg', (df['weight'] * 0.453592).astype(int))
#         ↑ position       ↑ name       ↑ values
```

### 7.2 Combining Columns

```python
# String concatenation
ufo['Location'] = ufo['City'] + ', ' + ufo['State']
```

### 7.3 Dropping Columns

```python
# Drop single column
df.drop('name', axis=1, inplace=True)
#              ↑ axis=1 means columns    ↑ modify in place

# Drop multiple columns
df.drop(['name', 'origin'], axis=1, inplace=True)

# Alternative: keep only specific columns
df = df[['mpg', 'cylinders', 'horsepower', 'weight']]
```

### 7.4 Renaming Columns

```python
df.rename(columns={'mpg': 'miles_per_gallon', 'cyl': 'cylinders'}, inplace=True)
```

### 7.5 Changing Data Types

```python
df['horsepower'] = df['horsepower'].astype(float)
df['cylinders'] = df['cylinders'].astype(int)
```

### Documentation:

- [drop()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html)
- [rename()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rename.html)

### Practice:

```python
df = pd.read_csv('data/auto-mpg.csv', na_values=['NA', '?'])

# Try:
# 1. Add a column 'weight_kg' = weight * 0.453592
# 2. Drop the 'name' column
# 3. Rename 'mpg' to 'miles_per_gallon'
```

---

## Step 8: Feature Normalization (Z-Score)

### Why Normalize?

Neural networks work better when features are on similar scales. A feature ranging 0-5000 will dominate one ranging 0-1.

### 8.1 Z-Score Normalization

Formula: `z = (x - mean) / std`

- Result: mean = 0, std = 1
- Values typically between -3 and +3
- Values beyond ±3 are outliers

```python
from scipy.stats import zscore

# Normalize a single column
df['mpg'] = zscore(df['mpg'])

# Or manually
mean = df['mpg'].mean()
std = df['mpg'].std()
df['mpg'] = (df['mpg'] - mean) / std
```

### 8.2 Helper Function for Z-Score

```python
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()
    if sd is None:
        sd = df[name].std()
    df[name] = (df[name] - mean) / sd

# Usage
encode_numeric_zscore(df, 'mpg')
encode_numeric_zscore(df, 'weight')
```

### 8.3 Min-Max Normalization (Alternative)

Scales to a range like 0-1 or -1 to 1:

```python
def encode_numeric_range(df, name, low=0, high=1):
    data_low = df[name].min()
    data_high = df[name].max()
    df[name] = ((df[name] - data_low) / (data_high - data_low)) * (high - low) + low

# Usage
encode_numeric_range(df, 'mpg', 0, 1)
```

### Documentation:

- [scipy.stats.zscore](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html)

---

## Step 9: Encoding Categorical Data

Neural networks need numbers, not text. Two main approaches:

### 9.1 Label Encoding

Convert categories to integers: `red → 0, green → 1, blue → 2`

```python
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
df['species'] = le.fit_transform(df['species'])
# Iris-setosa → 0, Iris-versicolor → 1, Iris-virginica → 2
```

**Use for:** Target variable (what you're predicting)

### 9.2 One-Hot Encoding (Dummy Variables)

Convert categories to binary columns:

```
species → species_setosa, species_versicolor, species_virginica
red     → [1, 0, 0]
green   → [0, 1, 0]
blue    → [0, 0, 1]
```

```python
# Using pd.get_dummies()
df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
# Creates: Sex_female, Sex_male, Embarked_C, Embarked_Q, Embarked_S
```

**Use for:** Input features (predictors)

### 9.3 Helper Functions from Lab

```python
# Label Encoding
def encode_text_index(df, name):
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

# One-Hot Encoding
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
```

### When to Use Which?

|Encoding|Use For|Example|
|---|---|---|
|Label|Target variable|Predicting species: 0, 1, 2|
|One-Hot|Input features|Color feature: red, green, blue|

### Documentation:

- [pd.get_dummies()](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
- [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)

###  Practice:

```python
train = pd.read_csv('data/titanic_train.csv')

# Try:
# 1. One-hot encode the 'Sex' column
# 2. One-hot encode the 'Embarked' column
# 3. Check the new columns created
```

---

## Step 10: Preparing Data for TensorFlow

### 10.1 The Goal

TensorFlow/PyTorch need:

- **X**: Feature matrix (NumPy array, shape: `[samples, features]`)
- **y**: Target vector (NumPy array)

### 10.2 Converting DataFrame to NumPy

```python
# Get values as NumPy array
X = df[['feature1', 'feature2', 'feature3']].values
y = df['target'].values

print(type(X))  # numpy.ndarray
```

### 10.3 The `to_xy` Helper Function

```python
def to_xy(df, target):
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
```

### 10.4 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X, y = to_xy(df, 'species')

# Split: 75% train, 25% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25, 
    random_state=42    # for reproducibility
)

print(X_train.shape)  # (112, 4)
print(X_test.shape)   # (38, 4)
print(y_train.shape)  # (112, 3)  ← one-hot encoded
print(y_test.shape)   # (38, 3)
```

### 10.5 Complete Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load data
df = pd.read_csv('data/iris.csv', na_values=['NA', '?'])

# 2. Handle missing values
df['petal_w'] = df['petal_w'].fillna(df['petal_w'].median())

# 3. Encode target (label encoding for classification)
encode_text_index(df, 'species')

# 4. Create X, y
X, y = to_xy(df, 'species')

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Ready for TensorFlow!
```

###  Documentation:

- [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

---

## Step 11: Practice Exercises

### Exercise 1: Load and Explore

```python
# Load 'data/titanic_train.csv'
# 1. How many rows and columns?
# 2. What are the column names?
# 3. How many missing values per column?
# 4. What's the average age?
# 5. How many survived vs died?
```

### Exercise 2: Clean the Data

```python
# Using titanic_train.csv:
# 1. Fill missing 'Age' with median
# 2. Fill missing 'Embarked' with most common value
# 3. Drop the 'Cabin' column (too many missing)
# 4. Drop 'Name', 'Ticket', 'PassengerId' (not useful for ML)
```

### Exercise 3: Encode Categories

```python
# Using cleaned titanic data:
# 1. One-hot encode 'Sex'
# 2. One-hot encode 'Embarked'
# 3. Verify new columns exist
```

### Exercise 4: Normalize Features

```python
# Using cleaned titanic data:
# 1. Z-score normalize 'Age'
# 2. Z-score normalize 'Fare'
```

### Exercise 5: Prepare for ML

```python
# 1. Create X, y (target is 'Survived')
# 2. Split into train/test (75/25)
# 3. Print shapes of X_train, X_test, y_train, y_test
```

### Exercise 6: Full Pipeline

```python
# Using 'data/iris.csv':
# 1. Load data
# 2. Check for missing values
# 3. Encode 'species' using label encoding
# 4. Create X, y
# 5. Split 80/20
# 6. Print final shapes
```

---

## Datasets Used in Lab 1

|File|Description|Columns|
|---|---|---|
|chipotle.tsv|Restaurant orders|order_id, quantity, item_name, choice_description, item_price|
|u.user|Movie reviewers|user_id, age, gender, occupation, zip_code|
|ufo.csv|UFO sightings|City, Colors Reported, Shape Reported, State, Time|
|auto-mpg.csv|Car fuel efficiency|mpg, cylinders, displacement, horsepower, weight, acceleration, year, origin, name|
|iris.csv|Flower measurements|sepal_l, sepal_w, petal_l, petal_w, species|
|titanic_train.csv|Titanic survival|PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked|

---

## Quick Reference Links

|Topic|Documentation|
|---|---|
|Pandas Main Docs|https://pandas.pydata.org/docs/|
|10 Min to Pandas|https://pandas.pydata.org/docs/user_guide/10min.html|
|read_csv|https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html|
|Indexing|https://pandas.pydata.org/docs/user_guide/indexing.html|
|Missing Data|https://pandas.pydata.org/docs/user_guide/missing_data.html|
|get_dummies|https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html|
|Sklearn train_test_split|https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html|


---

## How Lab 1 Connects to Lab 0

|NumPy (Lab 0)|Pandas (Lab 1)|
|---|---|
|`np.array()`|`df.values` (converts to NumPy)|
|`arr.shape`|`df.shape`|
|`arr[0:5]`|`df.iloc[0:5]`|
|`arr[:, 0]`|`df['column_name']`|
|`np.mean(arr)`|`df['col'].mean()`|
|Manual operations|Higher-level methods|

Pandas uses NumPy under the hood. When you extract data for ML, it becomes NumPy arrays!

---

Completed