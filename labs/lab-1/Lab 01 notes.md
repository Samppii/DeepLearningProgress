# Data Preprocessing with Pandas
## CSC 296S - Deep Learning | Dr. Victor Chen
**Official Docs:** [Pandas Documentation](https://pandas.pydata.org/docs/)

---

## Step 1: What is Pandas & Why Do We Need It?

### The Big Picture
NumPy = great for numerical arrays, Pandas = great for tabular data(rows and columns, like spreadsheets/CSVs)

```python
import pandas as pd # standard alias
import numpy as np
```

### Two Main Data Structures

| Structure | What It IS       | Like...           |
| --------- | ---------------- | ----------------- |
| Series    | 1D labeled array | A single column   |
| DataFrame | 2D labeled table | A spreadsheet/CSV |

```python
# Series - One Column
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

### Documentation

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
df = pd.read_csv('data/auto-mpg.csv', na_values = ['NA', '?'])
#                                     ↑ treat these as missing values
```

### 2.2 Reading Files with Different Separators

```python
# Tab-separated file (.tsv)
orders = pd.read_csv('data/chipotle.tsv', sep='\t')

# Pipe-separated file
users = pd.read_csv('data/u.user', sep = '|', header = None, names = ['user_id', 'age', 'gender', 'occupation', 'zip_code'])
```

### 2.3 Quick Data Inspection

```python
df.head() # First 5 rows
df.head(10) # First 10 rows
df.tail() # Last 5 rows
df.shape() # Rows, Columns
df.columns() # Column names
df.dtypes() # Data type of each column
df.info() # Summary Info
df.describe() # Statistics for numeric columns
```

### Documentation:

- [pd.read_csv()](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)

---

## Step 3: DataFrame Basics

### 3.1 What's in a DataFrame?

```python
df = pd.read_csv('data/auto-mpg.csv')

# See the structure
print(type(df)) # pandas.core.frame.DataFrame
print(df.shape) # (398, 9) - 398 rows, 9 columns
print(df.columns) # Index (['mpg', 'cylinders', ...])
```

### 3.2 Data Types

```python
df.dtypes
# mpg float64 
# cylinders int64 
# displacement float64 
# horsepower object ← string! might need cleaning 
# weight int64 
# ...`
```

#### Common dtypes:

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

## Step 4: Selecting Data (Series & Column)

### 4.1 Selecting a Single Column (Returns Series)

```python
# Two ways - both return a Series
df['mpg'] # Bracket notation (always works)
df.mpg # Dot notation (only for simple column names)
```

### 4.2 Selecting Multiple Columns (Returns DataFrame)

```python
# Pass a list of column names
df[['mpg','cylinders','horsepower']]
```

### 4.3 Selecting Rows by Index Position: `iloc`

```python
# iloc = integer location (position-based)
df.iloc[0] # First row
df.iloc[0:5] # First five rows
df.iloc[-1] # Last row
df.iloc[0, 2] # Row 0, Column 2
df.iloc[0:5, 0:3] # First 5 rows, First 3 columns
```

### 4.4 Selecting Rows by Label: `loc`

```python
# loc = label-based
df.loc[0] # Row with index label 0
df.loc[0:5] # Rows 0 through 5 (inclusive!)
df.loc[0, 'mpg'] # Row 0, Column 'mpg'
df.loc[0:5, 'mpg':'weight'] # Rows 0-5, columns mpg through weight
```


### Key Difference: `iloc` vs `loc`

|         | iloc                           | loc                              |
| ------- | ------------------------------ | -------------------------------- |
| Uses    | Integer Position               | Label / Name                     |
| Slicing | Exclusive end                  | Inclusive end                    |
| Example | `df.iloc[0:3]` -> rows 0, 1, 2 | `df.loc[0:3]` -> rows 0, 1, 2, 3 |

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
- [loc](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.loc.html)`

---

## Step 5: Handling Missing Values

### 5.1 Finding Missing Values

```python
# Check for missing values
df.isnull() # DataFrame of True/False
df.isnull().sum() # Count missing per column
df.isnull().sum().sum() # Total missing values

# Find rows with missing values in a specific column
df[df['horsepower'].isnull()]
```

### 5.2 Dropping Missing Values

```python
# Drop all rows with ANY missing values
df_clean = df.dropna()

# Drop rows where spicific column is missing
df_clean = df.dropna(subset = ['horsepower'])
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
# Tell Pandas What Counts as "missing"
df = pd.read_csv('../data/auto-msg.csv', na_values = ['NA', '?', 'NA',''])
```

### Documentation

- [Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html) 
- [fillna()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html) 
- [dropna()](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)

---

## Step 6: Sorting & Filtering

### 6.1 Sorting