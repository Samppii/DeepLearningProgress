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


