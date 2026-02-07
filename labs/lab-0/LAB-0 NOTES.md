# Lab 0: NumPy Fundamentals for Deep Learning
> **Course:** CSC 296S Deep Learning (Spring 2026)   
> **Official Docs:** [NumPy Documentation](https://numpy.org/doc/stable/)

---

## Study Overview

| Step | Topic |
|------|-------|
| 1 | Environment Setup | 
| 2 | What is NumPy & Why |
| 3 | Creating Arrays |
| 4 | Array Properties |
| 5 | Indexing & Slicing | 
| 6 | Array Operations | 
| 7 | Aggregation Functions & Axis |
| 8 | Reshaping Arrays |
| 9 | Random Number Generation |
| 10 | Practice Exercises |

---

## Step 1: Environment Setup

### What to install:
- **Anaconda**: [Download here](https://www.anaconda.com/download/)
- Includes: Python, NumPy, Pandas, Jupyter Notebook, and more

### Verify your installation:

```python
import numpy as np
import tensorflow as tf
import torch

print(f"NumPy Version: {np.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"PyTorch version: {torch.__version__}")
```
### Documentation:
- [Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/)
- [NumPy Installation](https://numpy.org/install/)

---


## Step 2: What is NumPy and Why do we need it?

### The problem with Python Lists:

```python
# Python list - VERY SLOW for math
python_list = [1,2,3,4,5]

# To multiply each element by 2, you need a loop
result = [x * 2 for x in python_list]
# [2,4,6,8,10]
```

### The NumPy solution:
```python
import numpy as np

# NumPy array = FAST for math and NO loop required
numpy_array = np.array([1,2,3,4,5])

# To Multiply each element of the array by 2, loop is not required
result = numpy_array * 2 # array([2,4,6,8,10])
```

### Why NumPy is faster:
1. **Written in C/Fortran** - Low-level Optimization
2. **Contiguous Memory** - Data stored in a block, not scattered
3. **Vectorized Operations** - Operations on entire arrays at once
4. **Homogeneous data** - All elements same type(no type checking overhead)

### Key Terminology:
| Term | Meaning |
|------|---------|
| **Array** | NumPy's main data structure (like a list, but faster) |
| **ndarray** | N-dimensional array (the actual class name) |
| **Vector** | 1D array, shape like `(n,)`|
| **Matrix** | 2D array, shape like `(m, n)`|
| **Tensor** | N-dimensional array(general term used in deep learning|

### Documentation:
- [What is NumPy?](https://numpy.org/doc/stable/user/whatisnumpy.html)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)

---

## Step 3: Creating Arrays

### 3.1 From Python lists

```python
import numpy as np

# 1D array (vector)
v = np.array([1,2,3,4])
print(v) # [1,2,3,4]
print(type(v)) # <class 'numpy.ndarray'>
print(v.shape) # (4,)      ← 1D, 4 elements


# 2D array (matrix) - List of Lists
M = np.array([[1, 2],
              [3, 4]])
print(M)
# [[1 2]
#  [3 4]]

# 3D array - List of List of Lists
T = np.array([[[1,2], [3,4]],
              [[5,6], [7, 8]]])
```

### 3.2 Using Built-in Functions

```python
# np.zeros() - array filled with 0s
zeros = np.zeros((3, 4))  # 3 rows, 4 columns of zeros

# np.ones() - array filled with 1s
ones = np.ones((2, 3)) # 2 rows, 3 columns of ones

# np.full() = array filled with specific value
fives = npm.full((2, 2), 5) # 2 rows, 2 columns filled with 5

# np.eye() - identity matrix (1s on diagonal)
identity = np.eye(3) # 3x3 identity matrix

# np.arrange() - like Python's range()
a = np.arrange(0, 10, 2) # [0, 2, 4, 6, 8] - start, stop, step

# np.linspace() - evenly spaced numbers
b = np.linspace(0, 1, 5) # [0, 0.25, 0.5, 0.75, 1] - start, stop, num_points
```

### 3.3 Specifying Data Type

```python
# Default is usually int64 or float64
arr_int = np.array([1,2,3])
print(arr_int.dtype) # int64

# Specifying the dtype explicitly
arr_int = np.array([1,2,3], dtype='float32'])
print(arr_float.dtype) # float32

# Common dtypes in deep learning:
# - float32: Most common for neural networks
# - float64: More precision, but slower
# - int32, int64: for indices, labels
```

### Documentation: 
- [np.array()](https://numpy.org/doc/stable/reference/generated/numpy.array.html)
- [np.zeros()](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)
- [np.ones()](https://numpy.org/doc/stable/reference/generated/numpy.ones.html)
- [np.arange()](https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
- [np.linspace()](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html)
- [Data types](https://numpy.org/doc/stable/user/basics.types.html)

### Practice:
```python
# 1. Create a 1D array of numbers 1-10
# 2. Create a 3x3 matrix of zeros
# 3. Create a 4x4 identity matrix
# 4. Create an array from 0 to 1 with 11 evenly spaced points
# 5. Create a 2x3 array of ones with dtype float32
```

---

## Step 4: Array Properties

### Essential Properties

```python
arr = np.array([[1,2,3,4],
               [5,6,7,8],
               [9,10,11,12]])

# .shape - dimensions of the array (Most Important)
print(arr.shape) #(3, 4) - 3 rows, 4 columns

# .ndim - number of dimensions
print(arr.ndim) # 2 (it's a 2D array)

# .size - total number of elements
print(arr.size) # 12 (3 * 4 = 12 elements)

# .dtype - data type of the elements
print(arr.dtype) # int64 by default

# len() - length of fist dimension
print(len(arr))
```

### Understanding Shape

```python
# 1D array
v = np.array([1,2,3])
print(v.shape) #(3,) - just 3 elements, no second dimensions

# 2D array
M = np.array([[1,2,3],
              [4,5,6]])
print(M.shape) #(2,3) - 2 rows, 3 columns

# 3D array
T = np.zeros((2,3,4))
print(T.shape)
```

### Why shape matters in Deep Learning?:
- Input data must match expected shape
- Matrix multiplication requires compatible shapes
- Shape mismatches cause most beginner errors!

### Documentation:
- [Array attributes](https://numpy.org/doc/stable/reference/arrays.ndarray.html#array-attributes)

### Practice:
```python
# Create arrays and predict their shape before checking:
a = np.zeros((5,3)) # 5 rows and 3 columns of zeros
b = np.array([1,2,3,4]) # 1D array of 1,2,3,4
c = np.ones((2,4,3)) # 3D array, think of it as 2 layers of 4x3
d = np.arrange(12) # Shape is (12,) - just a 1D array of 12 elements: [1,2,3,4,5,6,7,8,9,10,11,12]
```

## Step 5: Indexing and Slicing (Critical Skill)

### 5.1 Basic Indexing (Single Elements)

```python
# 1D array indexing
v = np.array([10,20,30,40,50])

v[0] # 10 - First element
v[2] # 20 - Third element
v[-1] # 50 - Last element
v[-2] # 40 - Second to last element

# 2D array indexing: arr[row, col]
M = np.array([[1,2,3],
              [4,5,6],
              [7,8,9]])
M[0,0] # Top left
M[0,2] # Top right
M[2,0] # Bottom left
M[2,2] # Bottom right
M[-1,-1] #9, last row last column
```
### 5.2 Slicing (Multiple Elements)

**Syntax:** `arr[start:stop:step]`
- `start`: Beginning index(inclusive, default to 0)
- `stop`: Ending index (exclusive, default end)
- `step`: Step size (default is 1)

```python
v = np.array([0,1,2,3,4,5,6,7,8,9])

v[2:5] # [2,3,4] - index 2 to 4
v[:4] # [0,1,2,3] - first 4 elements
v[6:] # [6,7,8,9] - from index 6 to end
v[::2] # [0,2,4,6,8] - every 2nd element
v[::-1] # [9,8,7,6,5,4,3,2,1,0] - reversed
v[1:8:2] # [1,3,5,7] - index 1 to 7, step 2
```

### 5.3 2D Slicing (Rows and Columns)

```python
M = np.array([[ 0,1,2,3],
              [10,11,12,13],
              [20,21,22,23],
              [30,31,32,33]])

# Syntax: arr[row_slice, col_slice]

# Get entire row
M[0, :] # [0, 1, 2, 3] - First row
M[1, :] # [10, 11, 12, 13] - Second row
M[-1, :] # [30, 31, 32, 33] - Last row

# Get entire column
M[:, 0] # [0, 10, 20, 30] - First Column
M[:, -1] # [3, 13, 23, 33] - Second Column

# Get submatrix (block)
M[0:2, 0:2] # [[0, 1],
            #  [10, 11]] - top-left 2x2

M[1:3, 2:4] # [[12,13],
            #  [22, 23]] - middle-right 2x2

# Skip rows/columns
M[::2, :] # Every other rows
M[:, ::2] # Every other column
```
### 5.4 Fancy Indexing (Using Arrays as Indices)

```python
arr = np.array([10,20,30,40,50])

# Use a list of indices
indices = [0, 2, 4]
arr[indices] #[10, 30, 50]

# 2D Fancy Indexing
M = np.array([[ 0, 1, 2],
              [10, 11, 12],
              [20, 21, 22]])
row_indices = [0, 2]
M[row_indices, :] #[[0, 1, 2],
                  # [20, 21, 22]]
```

### 5.5 Boolean Indexing (Filtering)

```python
arr = np.array([1,5,3,8,2,9,4])

# Creating boolean mask
mask = arr > 4
print(mask) # [False, True, False, True, False, True, False]

# Use mask to filter
arr[mask] # [5, 8, 9]

# Or directly:
arr[arr > 4] # [5, 8, 9]
arr[arr % 2 == 0] # [8, 2, 4] - even numbers
```

### 5.6 Important: Views vs Copies

```python
# Slices create VIEWS (share memory with original!)
original = np.array([1, 2, 3, 4, 5])
slice_view = orginal[1:4]

slice_view[0] = 999
print(original) # [1, 999, 3, 4, 5] - ORIGINAL CHANGED!


# To avoid this, make a copy:
slice_copy = original[1:4].copy
slice_copy[0] = 888
print(original) # [1, 999, 3, 4, 5] - Original Unchanged
```

### Documentation:
- [Indexing basics](https://numpy.org/doc/stable/user/basics.indexing.html)
- [Advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)


### Practice:
```python
M = np.array([[ 0,  1,  2,  3,  4],
              [10, 11, 12, 13, 14],
              [20, 21, 22, 23, 24],
              [30, 31, 32, 33, 34],
              [40, 41, 42, 43, 44]])

# Try to get:
# 1. The element at row 2, column 3
# 2. The entire third row
# 3. The entire second column
# 4. A 3x3 block from the center
# 5. Every other row
# 6. All elements greater than 25
```

---

## Step 6: Array Operations (Vectorization)

### 6.1 Scalar Operations

```python
arr = np.array([1,2,3,4,5])

arr + 10 # [11,12,13,14,15]
arr - 5 # [-4,-3,-2,-1,0]
arr * 2 # [2,4,6,8,10]
arr / 2 # [0.5,1.0,1.5,2.0,2.5]
arr ** 2 # [1,4,9,16,25]
arr % 2 # [1,0,1,0,1] - modulo
```

### 6.2 Element-wise Operations

```python
a = np.array([1,2,3])
b = np.array([4,5,6])

a + b # [5,7,0]
a - b # [-3, -3, -3]
a * b # [4, 10, 18] - element-wise, NOT matrix multiplication!
a / b # [0.25, 0.4, 0.5]
a ** b # [1, 32, 729] - 1^4, 2^5, 3^6
```

### 6.3 Matrix Multiplication

```python
A = np.array([[1, 2],
             [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

# Element-wise multiplication (Hadamard product)
A * B # [[5, 12],
      # [7, 8]]

# Matrix multiplication - THREE ways:
np.dot(A, B) # [[19, 22],
             # [43, 50]]

A @ B # Same as above (Python 3.5+)

np.matmul(A, B) # Same as above
```

### Matrix Multiplication Shape Rules:

```
(m, n) @ (n, p) = (m, p)
          ^  ^
          Must match!
```

```python
# Example:
A = np.ones((3, 4)) # 3x4
B = np.ones((4, 2)) # 4x2
C = A @ B

# This would fail
# A = np.ones((3, 4)) # 3x4
# B = np.ones((3, 2)) # 3x2
# C = A @ B # Error! 4 != 3
```

### 6.4 Universal Functions (ufuncs)

```python
arr = np.array([0, np.pi/2, np.pi])

# Trigonometric
np.sin(arr) # [0, 1, 0]
np.cos(arr) # [1, 0, -1]

# Exponential and logarithmic
np.exp(np.array([1, 2, 3])) # [e^1, e^2, e^3]
np.log(np.array([1, np.e, np.e**2])) # [1, 3]
```

### 6.5 Broadcasting (Advanced but Important)

NumPy can operate on arrays of different shapes in certain cases:

```python
# Scalar broadcasts to all elements
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
arr + 10 # Adds 10 to every element

# 1D array broadcasts across rows
row = np.array([10, 20, 30])
arr + row # [[11, 22, 33],
          #  [14, 25, 36]]

# Column vector broadcasts across columns
col = np.array([[100],
                [200]])
arr + col # [[101, 102, 103],
          #  [204, 205, 206]]
```

### Documentation:
- [Universal functions](https://numpy.org/doc/stable/reference/ufuncs.html)
- [Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [np.dot()](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)

### Practice:
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Calculate:
# 1. A + B
# 2. A * B (element-wise)
# 3. A @ B (matrix multiplication)
# 4. A squared (each element squared)
# 5. Square root of each element in B
```

---

## Step 7: Aggregation Functions & The Axis Parameter

### 7.1 Basic Aggregations

```python
arr = np.array([1, 2, 3, 4, 5])

np.sum(arr)       # 15
np.prod(arr)      # 120 (1*2*3*4*5)
np.mean(arr)      # 3.0
np.std(arr)       # 1.414... (standard deviation)
np.var(arr)       # 2.0 (variance)
np.min(arr)       # 1
np.max(arr)       # 5
np.argmin(arr)    # 0 (index of minimum)
np.argmax(arr)    # 4 (index of maximum)
```

### 7.2 The `axis` Parameter (VERY IMPORTANT!)

For 2D arrays:
- `axis=None` (default): Operation on ALL elements
- `axis=0`: Operation along columns (collapses rows)
- `axis=1`: Operation along rows (collapses columns)

**Memory trick:** The axis you specify is the one that DISAPPEARS.

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])
# Shape: (2, 3)

# No axis - sum everything
np.sum(M)              # 21

# axis=0: collapse rows (go DOWN), result shape (3,)
np.sum(M, axis=0)      # [5, 7, 9]
#                         1+4, 2+5, 3+6

# axis=1: collapse columns (go ACROSS), result shape (2,)
np.sum(M, axis=1)      # [6, 15]
#                         1+2+3, 4+5+6
```

### Visual Representation:

```
        axis=1 →
       ┌─────────┐
axis=0 │ 1  2  3 │ → sum = 6
   ↓   │ 4  5  6 │ → sum = 15
       └─────────┘
         ↓  ↓  ↓
         5  7  9
```

### More Examples:

```python
M = np.array([[1, 5, 3],
              [4, 2, 6]])

# Mean
np.mean(M)            # 3.5 (all elements)
np.mean(M, axis=0)    # [2.5, 3.5, 4.5] (mean of each column)
np.mean(M, axis=1)    # [3.0, 4.0] (mean of each row)

# Max
np.max(M)             # 6
np.max(M, axis=0)     # [4, 5, 6] (max in each column)
np.max(M, axis=1)     # [5, 6] (max in each row)
```

### Documentation:
- [np.sum()](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)
- [np.mean()](https://numpy.org/doc/stable/reference/generated/numpy.mean.html)
- [Statistical functions](https://numpy.org/doc/stable/reference/routines.statistics.html)

### Practice:
```python
M = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

# Calculate:
# 1. Sum of all elements
# 2. Sum of each column
# 3. Sum of each row
# 4. Mean of each row
# 5. Max of each column
# 6. Index of max in each row (use argmax)
```

---

## Step 8: Reshaping Arrays

### 8.1 reshape()

```python
arr = np.arange(12)   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# Reshape to 2D
arr.reshape(3, 4)     # 3 rows, 4 columns
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

arr.reshape(4, 3)     # 4 rows, 3 columns
# [[ 0,  1,  2],
#  [ 3,  4,  5],
#  [ 6,  7,  8],
#  [ 9, 10, 11]]

# Use -1 to auto-calculate one dimension
arr.reshape(3, -1)    # 3 rows, auto columns → (3, 4)
arr.reshape(-1, 6)    # auto rows, 6 columns → (2, 6)

# Reshape to 3D
arr.reshape(2, 2, 3)  # 2 "layers" of 2x3 matrices
```

### 8.2 flatten() and ravel()

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])

# flatten() - returns a COPY
flat = M.flatten()    # [1, 2, 3, 4, 5, 6]

# ravel() - returns a VIEW (more memory efficient)
raveled = M.ravel()   # [1, 2, 3, 4, 5, 6]
```

### 8.3 transpose()

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])
# Shape: (2, 3)

M.T                   # Transpose
# [[1, 4],
#  [2, 5],
#  [3, 6]]
# Shape: (3, 2)

# Also works:
np.transpose(M)
M.transpose()
```

### 8.4 Adding Dimensions

```python
arr = np.array([1, 2, 3])   # Shape: (3,)

# Add dimension using np.newaxis or None
arr[np.newaxis, :]          # Shape: (1, 3) - row vector
arr[:, np.newaxis]          # Shape: (3, 1) - column vector

# Using reshape
arr.reshape(1, -1)          # Shape: (1, 3)
arr.reshape(-1, 1)          # Shape: (3, 1)

# Using np.expand_dims
np.expand_dims(arr, axis=0) # Shape: (1, 3)
np.expand_dims(arr, axis=1) # Shape: (3, 1)
```

### Documentation:
- [np.reshape()](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html)
- [np.transpose()](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html)
- [np.expand_dims()](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)

### Practice:
```python
arr = np.arange(24)

# Reshape to:
# 1. (6, 4)
# 2. (2, 3, 4)
# 3. (4, -1) - let NumPy figure out second dimension
# 4. Flatten a 3x4 matrix back to 1D
# 5. Transpose a 2x5 matrix
```

---

## Step 9: Random Number Generation

### 9.1 Basic Random Functions

```python
from numpy import random

# Random floats between 0 and 1
random.rand(3)           # 1D array of 3 random numbers
random.rand(2, 3)        # 2x3 array of random numbers

# Random floats from standard normal distribution (mean=0, std=1)
random.randn(3)          # 1D array
random.randn(2, 3)       # 2x3 array

# Random integers
random.randint(0, 10, size=5)      # 5 random ints from 0-9
random.randint(0, 10, size=(2, 3)) # 2x3 array of random ints
```

### 9.2 Setting Random Seed (for reproducibility)

```python
np.random.seed(42)       # Set seed

# Now random numbers will be the same every time
arr1 = np.random.rand(3)
print(arr1)              # Always same values

np.random.seed(42)       # Reset seed
arr2 = np.random.rand(3)
print(arr2)              # Same as arr1!
```

### 9.3 Other Distributions

```python
# Uniform distribution between a and b
random.uniform(low=0, high=10, size=5)

# Normal distribution with custom mean and std
random.normal(loc=5, scale=2, size=5)  # mean=5, std=2

# Random choice from array
random.choice([1, 2, 3, 4, 5], size=3)           # With replacement
random.choice([1, 2, 3, 4, 5], size=3, replace=False)  # Without

# Shuffle array in place
arr = np.array([1, 2, 3, 4, 5])
random.shuffle(arr)
```

### Documentation:
- [Random sampling](https://numpy.org/doc/stable/reference/random/index.html)
- [np.random.rand()](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html)
- [np.random.randn()](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html)

---

