# NumPy Cheatsheet

> Quick reference for Lab 0

```python
import numpy as np
```

---

## Creating Arrays

```python
np.array([1, 2, 3])              # from list
np.zeros((3, 4))                 # 3x4 of zeros
np.ones((2, 3))                  # 2x3 of ones
np.full((2, 2), 5)               # 2x2 filled with 5
np.eye(3)                        # 3x3 identity matrix
np.arange(0, 10, 2)              # [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)             # 5 evenly spaced from 0 to 1
np.random.rand(3, 3)             # 3x3 random (0-1)
np.random.randn(3, 3)            # 3x3 random (normal dist)
np.random.randint(0, 10, (2,3))  # 2x3 random integers 0-9
```

---

## Array Properties

```python
arr.shape      # dimensions (rows, cols)
arr.size       # total elements
arr.dtype      # data type
arr.ndim       # number of dimensions
```

---

## Indexing & Slicing

```python
# 1D
arr[0]         # first element
arr[-1]        # last element
arr[1:4]       # indices 1, 2, 3

# 2D
M[0, 0]        # row 0, col 0
M[1, :]        # entire row 1
M[:, 2]        # entire column 2
M[0:2, 0:2]    # top-left 2x2 block
M[-1]          # last row
```

**Remember:** `start:stop` → stop is excluded

---

## Slicing Syntax

```python
arr[start:stop:step]

arr[::2]       # every 2nd element
arr[::-1]      # reversed
arr[:3]        # first 3
arr[3:]        # from index 3 to end
```

---

## Operations

```python
# Scalar (applies to all elements)
arr + 10
arr * 2
arr ** 2

# Element-wise (same shape arrays)
a + b
a * b          # NOT matrix multiplication

# Matrix multiplication
np.dot(A, B)
A @ B          # same thing
```

---

## Aggregations

```python
arr.sum()
arr.mean()
arr.std()
arr.min()
arr.max()
arr.argmin()   # index of min
arr.argmax()   # index of max
```

### Axis Parameter

```python
# 2D array:
M.sum()           # sum all
M.sum(axis=0)     # sum each column (collapse rows)
M.sum(axis=1)     # sum each row (collapse columns)
```

**Trick:** axis you specify is the one that disappears

---

## Reshaping

```python
arr.reshape(3, 4)      # reshape to 3x4
arr.reshape(-1, 4)     # auto-calculate rows
arr.flatten()          # to 1D (copy)
arr.ravel()            # to 1D (view)
arr.T                  # transpose
```

---

## Boolean Indexing

```python
arr[arr > 5]           # elements > 5
arr[arr % 2 == 0]      # even elements
```

---

## Useful Functions

```python
np.sqrt(arr)
np.exp(arr)
np.log(arr)
np.abs(arr)
np.round(arr)
np.sin(arr)
np.cos(arr)
```

---

## Views vs Copies

```python
# Slice = VIEW (shares memory)
b = a[0:3]
b[0] = 999      # changes a too!

# Make a copy
b = a[0:3].copy()
b[0] = 999      # a unchanged
```

---

## Set Seed (reproducibility)

```python
np.random.seed(42)
```