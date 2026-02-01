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
