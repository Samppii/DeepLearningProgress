# Lecture 3: Linear Classifier
## CSC 296S - Deep Learning | Dr. Victor Chen

---

## IMPORTANT
This is the foundational lecture that bridges classical ML to deep learning. Everything here - linear classifiers, softmax, cross-entropy, and stacking layers - is exactly what neural networks are built from.

**Key insight from the slides:**

```
Neural Network = Stack of Linear Classifiers + Non-linear Activations
```

---

## 1. Linear Classifier = The Building Block

In deep learning, a linear classifier is also called:
- **Feed-forward layer**
- **Fully Connected Layer (FC)**
- **Dense Layer**

It's just a matrix multiplication between input `x` and weights `W`.

**The Formula:**

```
f(x,W) = Wx + b
```

Where:
- `x` = input vector (your data)
- `W` = weight matrix (learned parameters)
- `b` = bias vector (offset term)
- `f(x,W)` = output scores fro each class

---

# 2. Matrix Multiplication Deep Dive
### 2.1 The Dimensions

For image classification with CIFAR-10 style images:

```
Input image: 32x32x3 = 3,072 pixels (flattened to column vector)
Output: 10 class scores

Dimensions:
[10×1] = [10×3072] × [3072×1] + [10×1]
s      =     W     ×    x     +   b
```

- `W` is a 10×3072 matrix (30,720 learnable weights!)
- Each **row** of W is a "template" for one class
- `b` is a 10×1 bias vector (one bias per class)

### 2.2 Simplified Example

**Setup:** 4-pixel image, 3 classes (cat/dog/ship)

```
Image (2×2):         Flattened:
┌────┬────┐          [56]
│ 56 │231 │    →     [231]
│ 24 │ 2  │          [24]
└────┴────┘          [2]
```

**The Computation:**

```
W                             x        b       s (scores)
┌─────┬─────┬─────┬─────┐   ┌────┐   ┌────┐   ┌───────┐
│ 0.2 │-0.5 │ 0.1 │ 2.0 │   │ 56 │   │1.1 │   │ -96.8 │ ← cat
│ 1.5 │ 1.3 │ 2.1 │ 0.0 │ × │231 │ + │3.2 │ = │ 437.9 │ ← dog
│ 0   │0.25 │ 0.2 │-0.3 │   │ 24 │   │-1.2│   │ 61.95 │ ← ship
└─────┴─────┴─────┴─────┘   │ 2  │   └────┘   └───────┘
                            └────┘
```

**Prediction:** Dog (highest score = 437.9)

---

## 3. The Problem: How Do We Know if W is Good?

Given some weights W, we get scores. But how do we measure if these scores are correct?

**Example:** 3 training images, 3 classes

| Image | True Label | Cat Score | Car Score | Frog Score |
| ----- | ---------- | --------- | --------- | ---------- |
| 🐱    | cat        | 3.2       | 5.1       | -1.7       |
| 🚗    | car        | 1.3       | 4.9       | 2.0        |
| 🐸    | frog       | 2.2       | 2.5       | -3.1       |
Problem: For the cat image, the car score (5.1) is higher than the cat score (3.2). This W is bad!

**We need:**
1. A **loss function** to quantify how wrong our predictions are
2. An **optimization method** to find better W (gradient descent)

---

## 4. From Scores to Probabilities: Softmax

Raw scores can be any number (positive, negative, huge, tiny). We want **probabilities** that:
- Are between 0 and 1
- Sum to 1 across all classes

### 4.1 The Softmax Function

```
P(class_i) = e^(score_i) / Σ e^(score_j)
```

**Two operations:**
1. **Exponentiate** - Makes everything positive, amplifies differences
2. **Normalize** - divide by sum so everything adds to 1

### 4.2 Worked Example

```
Raw scores:           exp():             Normalized (softmax):
cat: 3.2      →    e^3.2 = 24.5    →      24.5/188.68 = 0.13
car: 5.1      →    e^5.1 = 164.0   →      164.0/188.68 = 0.87
frog: -1.7    →    e^-1.7 = 0.18   →      0.18/188.68 = 0.0
                   ─────────────
                   Sum = 188.68           Sum = 1.00
```

**Interpretation:** Model thinks there's an 87% chance it's a car, 13% chance it's a cat.

### 4.3 Terminology

| Term                      | Meaning                                            |
| ------------------------- | -------------------------------------------------- |
| **Logits**                | Raw scores before softmax (can be any real number) |
| **Softmax Probabilities** | Output after softmax (0-1, sum to 1)               |
| **Log-probabilities**     | Another name for logits (because softmax uses exp) |

---

## 5. Cross-Entropy Loss

Now we can compare predicted probabilities to the true label.

### 5.1 The True Label as One-Hot

If the true class is "cat" (class 0 out of 3):

```
True distribution : [1.0, 0.0, 0.0]
					 cat  car  frog
```

### 5.2 Cross-Entropy Formula

```
Loss = -Σ y_true × log(y_pred)
```

Since y_true is one-hot (only one 1, rest are 0s), this simplifies to:

```
Loss = -log(P_predicted for correct class)
```

### 5.3 Worked Example (IMPORTANT)

**Predicted probabilities:** [0.13, 0.87, 0.00] (cat, car, frog)
**True Label:** cat → one-hot: [1, 0, 0]

```
Loss = -(1xlog(0.13) + 0xlog(0.87) + 0xlog(0.00))
	 = -log(0.13)
	 = 2.04
```

**If prediction was perfect (P(cat) = 1.0):**
```
Loss = -log(1.0) = 0 ← Perfect!
```

**If prediction was confident but not wrong** (P(cat) = 0.01):
```
Loss = -log(0.01) = 4.6 ← Very high penalty!
```

### 5.4 Key Properties of Cross-Entropy

- **Perfect prediction** → Loss = 0
- **Confident wrong prediction** → Loss is HUGE (Heavily penalized)
- **Uncertain prediction** → Moderate loss
- This is why neural networks learn to be confident on correct answers

### 5.5 Cross-Entropy vs KL Divergence

| Metric            | Use Case                                                           |
| ----------------- | ------------------------------------------------------------------ |
| **Cross-Entropy** | Supervised learning - comparing predictions to ground truth labels |
| **KL Divergence** | Unsupervised Learning - comparing two probability distributions    |

For classification with one-hot labels, they're mathematically related, but cross-entropy is the standard loss function.

---
