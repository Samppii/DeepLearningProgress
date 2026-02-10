# Pandas Cheatsheet

> Quick reference for Lab 1

```python
import pandas as pd
```

---

## Reading Data

```python
df = pd.read_csv('file.csv')                    # comma-separated
df = pd.read_csv('file.tsv', sep='\t')          # tab-separated
df = pd.read_csv('file.txt', sep='|')           # pipe-separated
df = pd.read_csv('file.csv', na_values=['?'])   # treat '?' as NaN
```

---

## Quick Inspection

```python
df.head()          # first 5 rows
df.head(10)        # first 10 rows
df.tail()          # last 5 rows
df.shape           # (rows, cols)
df.columns         # column names
df.dtypes          # data types
df.info()          # summary
df.describe()      # stats for numeric columns
```

---

## Selecting Columns

```python
df['name']                    # single column → Series
df.name                       # same (dot notation)
df[['name', 'age']]           # multiple columns → DataFrame
```

---

## Selecting Rows

### iloc - by position (integer)

```python
df.iloc[0]                    # first row
df.iloc[-1]                   # last row
df.iloc[0:5]                  # rows 0-4
df.iloc[5, 2]                 # row 5, column 2
df.iloc[0:5, 0:3]             # rows 0-4, columns 0-2
```

### loc - by label (name)

```python
df.loc[0]                     # row with index 0
df.loc[0:5]                   # rows 0-5 (inclusive!)
df.loc[0, 'name']             # row 0, column 'name'
df.loc[0:5, 'mpg':'weight']   # rows 0-5, columns mpg to weight
```

### Key difference

||iloc|loc|
|---|---|---|
|Uses|position|label|
|End|excluded|included|

---

## Boolean Filtering

```python
df[df['mpg'] > 30]                              # single condition
df[(df['mpg'] > 30) & (df['cylinders'] == 4)]   # AND
df[(df['mpg'] > 30) | (df['cylinders'] == 4)]   # OR
df[df['name'].str.contains('ford')]             # string contains
```

**Note:** Filtering selects **rows**, not columns

---

## Missing Values

```python
df.isnull()                   # True/False for each cell
df.isnull().sum()             # count missing per column

df.dropna()                   # drop rows with any NaN
df.dropna(subset=['col'])     # drop rows where 'col' is NaN

df['col'].fillna(0)           # fill with 0
df['col'].fillna(df['col'].median())   # fill with median
df['col'].fillna(df['col'].mean())     # fill with mean
```

---

## Sorting

```python
df.sort_values('col')                    # ascending
df.sort_values('col', ascending=False)   # descending
df.sort_values(['col1', 'col2'])         # multiple columns
```

---

## Adding / Modifying Columns

```python
df['new_col'] = df['a'] + df['b']        # new column
df['col'] = df['col'] * 2                # modify existing
df.drop('col', axis=1)                   # drop column (returns new df)
df.drop('col', axis=1, inplace=True)     # drop column (modifies df)
df.rename(columns={'old': 'new'})        # rename column
```

---

## Aggregations

```python
df['col'].sum()
df['col'].mean()
df['col'].median()
df['col'].std()
df['col'].min()
df['col'].max()
df['col'].value_counts()     # count unique values
```

---

## Z-Score Normalization

```python
from scipy.stats import zscore

df['col'] = zscore(df['col'])

# Or manually:
mean = df['col'].mean()
std = df['col'].std()
df['col'] = (df['col'] - mean) / std
```

---

## Encoding Categorical Data

### One-Hot Encoding (for features)

```python
df = pd.get_dummies(df, columns=['Sex', 'Color'])
# Sex → Sex_male, Sex_female
# Color → Color_red, Color_blue, Color_green
```

### Label Encoding (for target)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])
# setosa → 0, versicolor → 1, virginica → 2
```

---

## Convert to NumPy

```python
arr = df.values                          # entire df
arr = df['col'].values                   # single column
arr = df[['col1', 'col2']].values        # multiple columns
```

---

## Train/Test Split

```python
from sklearn.model_selection import train_test_split

X = df[['feature1', 'feature2']].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25,      # 25% for test
    random_state=42      # reproducibility
)
```

---

## Common Patterns

```python
# Load → Inspect → Clean → Encode → Split

# 1. Load
df = pd.read_csv('data.csv', na_values=['?'])

# 2. Inspect
df.head()
df.isnull().sum()

# 3. Clean
df['col'] = df['col'].fillna(df['col'].median())
df.drop(['useless_col'], axis=1, inplace=True)

# 4. Encode
df = pd.get_dummies(df, columns=['category_col'])

# 5. Convert & Split
X = df.drop('target', axis=1).values
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```

