# CSC 296S: Deep Learning

> **Course:** CSC 296S Deep Learning (Spring 2026)  
> **Institution:** California State University, Sacramento  
> **Instructor:** Dr. Haiquan Chen

Personal learning repository for the Deep Learning course. Contains lab work, notes, and practice exercises.

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/deep-learning-csc296s.git
cd deep-learning-csc296s

# Create virtual environment
python3 -m venv dl-env

# Activate it
source dl-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

---

## Repository Structure

```
deep-learning-csc296s/
│
├── README.md                 # You are here
├── requirements.txt          # Python dependencies
├── .gitignore
│
├── lab0/
│   ├── CSC296S_Lab0_numpy.ipynb    # Original lab notebook
│   ├── notes.md                     # My notes
│   └── practice.ipynb               # Extra practice
│
├── lab1/
│   ├── lab1.ipynb
│   └── notes.md
│
├── lab2-.../
│   └── ...
│
└── resources/
    └── cheatsheets/
```

---

## Progress Tracker

| Lab | Topic | Status | Date Completed |
|-----|-------|--------|----------------|
| Lab 0 | NumPy Fundamentals | 🟡  |
| Lab 1 | TBD | ⬜ Not Started | - |
| Lab 2 | TBD | ⬜ Not Started | - |
| Lab 3 | TBD | ⬜ Not Started | - |
| Lab 4 | TBD | ⬜ Not Started | - |
| Lab 5 | TBD | ⬜ Not Started | - |

**Legend:** ✅ Complete | 🟡 In Progress | ⬜ Not Started

---

## What I Learned

### Lab 0: NumPy Fundamentals
- Creating arrays (`np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`)
- Array properties (`.shape`, `.dtype`, `.size`)
- Indexing and slicing
- Vectorized operations
- Matrix multiplication (`@`, `np.dot`)
- Aggregations with `axis` parameter
- Reshaping arrays

### Lab 1: TBD
- Coming soon...

---

## Tools & Technologies

- **Python 3.13**
- **NumPy** - Array operations
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **TensorFlow** - Deep learning framework
- **PyTorch** - Deep learning framework
- **Jupyter Lab** - Interactive notebooks

---

## Resources

### Official Documentation
- [NumPy Docs](https://numpy.org/doc/stable/)
- [TensorFlow Docs](https://www.tensorflow.org/api_docs)
- [PyTorch Docs](https://pytorch.org/docs/stable/index.html)
- [Matplotlib Docs](https://matplotlib.org/stable/contents.html)

### Cheatsheets
- [NumPy Cheatsheet](https://numpy.org/doc/stable/user/cheatsheet.html)

---

## Environment Info

```
Python: 3.13
NumPy: latest
TensorFlow: latest
PyTorch: latest
```

To check versions:
```python
import numpy as np
import tensorflow as tf
import torch

print(f"NumPy: {np.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"PyTorch: {torch.__version__}")
```

---

## Notes

- All labs completed in Jupyter Lab
- Notes written in Obsidian, exported to Markdown
- Virtual environment (`dl-env/`) not included in repo - use `requirements.txt` to recreate

---

## Acknowledgments

- Dr. Haiquan Chen for the course material
- California State University, Sacramento

---

*Last updated: January 2026*
