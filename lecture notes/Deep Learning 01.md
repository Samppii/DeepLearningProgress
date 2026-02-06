# Lecture 1: "Deep Learning is All You Need"
## CSC 296S - Deep Learning | Dr. Victor Chen

--- 

## 1. The Big Picture: `AI → ML → DL`
Understanding the hierarchy is the foundation of everything. These are **not** the same thing - they are nested subfields.

```
Artificial Intelligence (AI)
└── Machine Learning (ML) 
	└── Deep Learning (DL)
```

- **AI** is the broadest field: Creating algorithm that let machines do tasks that normally require human intelligence. This includes planning, reasoning, knowledge representation, NLP, computer vision, robotics, and yes - Machine Learning.
- **Machine Learning** is a subfield of AI that specifically focuses on **learning from data** and making predictions on data it hasn't seen before. Instead of hand-coding rules, you let the algorithm figure out the patterns.
- **Deep Learning** is a subfield of ML that uses **deep neural networks** to automatically extract features from data. This is the focus of this course.

**Key Takeaway:** DL is not a replacement for ML or AI. It's a specific *technique* within ML that has become dominant because of its ability to handle unstructured data (images, text, audio) with superior accuracy.

---

## 2. Types of Machine Learning (Learning Paradigms)
These are the different ways a machine can "learn." 

### 2.1 Supervised Learning
- **How it works:** You give the model labeled data (input → correct output pairs). The model learns the mapping from input to output.
- **Examples:**
	- Image Classification: Given an image, predict the label (e.g., "cat" or "dog")
	- Email Classification: Spam or not spam
	- Regression: Predicting a real-valued output (e.g., house prices)
	- **This is the main focus of the early modules in this course.**

### 2.2 Unsupervised Learning
- **How it works:** No labels. The model finds hidden patterns or structure in the data on its own.
- **Example:** Clustering - grouping similar data points together without being told what the groups are.

### 2.3 Semi-Supervised Learning
- **How it works:** A mix - you have *some* labeled data and a *lot* of unlabeled data. The model uses both.
- **Example:** Generative AI models often leverage this paradigm.

### 2.4 Reinforcement Learning (RL)
- **How it works:** The model learns by interacting with an environment. It takes actions, gets rewards or penalties, and learns to maximize reward over time.
- **Example:** AlphaGo learning to play the board game Go.

### 2.5 Self-Supervised learning
- **How it works:** The model creates its own labels from the structure of the data iteself. No human labeling needed.
- **Example:** Language models predicting the next word in a sentence - the "label" is already in the data.

---

## 3. What Makes Deep Learning Special?

### 3.1 Automatic Feature Extraction
This is the **biggest deal** about DL compared to traditional ML.

- **Traditional ML:** You (the human) had to manually design features. For example, to classify a face, you'd hand0craft features like "edge detectors," "skin color histograms," etc. This is called **feature engineering** and it's tedious, error-prone, and doesn't really scale.
- **Deep Learning:** The network learns the features automatically from raw data. No manual feature engineering needed.

### 3.2 Hierarchical Feature Learning
Deep networks lean features at multiple levels of abstraction (this is actually where the name "deep" comes from - multiple layers):

```
Raw Input (e.g., pixels of an image) 
↓ 
Low-Level Features → edges, textures, colors 
↓
Mid-Level Features → shapes, parts (e.g., eyes, wheels) 
↓
High-Level Features → full objects (e.g., "car", "face")
↓
Final Classification → output label
```

This visualization comes from the famous paper by Zeiler & Fergus (2014) - "Visualizing and Understanding Convolutional Networks." They actually showed what each layer of a neural network was learning visually.

### 3.3 Handles Unstructured Data
DL shines on unstructured data like images, text, and audio - things that are hard to represent with hand-crafted features. Traditional ML struggles here; DL dominates.

---

## 4. Discriminative AI vs. Generative AI
Very important distinction you'l see throughout the course and in the real world.

|              | Discriminative Models                | Generative Models                                  |
| ------------ | ------------------------------------ | -------------------------------------------------- |
| **Goal**     | Classify / Predict labels            | Generate new synthetic data                        |
| **Asks**     | "How do I get y  given x?"           | "How can I get x?"                                 |
| **Examples** | Image classification, span detection | Image generation, text generation, voice synthesis |

- **Discriminative** = Takes input, give you a category/label. It learns the **boundary** between classes.
- **Generative** = Creates new data that looks like real data. It learns the **distribution** of data itself.

---

## 5. Specialized AI vs. Generalized AI

- **Specialized AI (Narrow AI):** Designed to do ONE specific task really well. Example:  a mode that plays chess, or one that classifies skin cancer images. Everything that I build during the labs will fall into this category.
- **Generalized AI (AGI)**: A hypothetical AI that can do *anything* a human can do - reason, learn, adapt across all domains. **AGI does not exist yet.** This is still an open research problem.

---

## 6. History of AI - The Timeline

Understanding the timeline gives context for *why* deep learning is where it is now:

| **Era**          | **What Happened**                                                                                                                                            |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Before 1980s** | Rule-based systems, uncertain/probabilistic reasoning. People tried to hand-code intelligence.                                                               |
| **1980s**        | Machine learning takes over. Algorithm that learn from data start to dominate.                                                                               |
| **2010s**        | Deep learning explodes. GPUs become powerful enough, data becomes abundant, and architectures like CNNs and RNNs start crushing benchmarks.                  |
| **2017**         | **Transformers** arrive ("Attentions Is All You Need" - Vaswani et al., 2017). This is the architecture behind GPT, BERT, and all modern LLMs. Game changer. |

**Why did DL take off in the 2010s?** Three things aligned:
1. **Data** - the internet gave us massive datasets
2. **Compute** - GPUs became cheap and powerful
3. **Algorithms** - better architectures and training techniques

---

## 7. Real-World Applications of Deep Learning

### 7.1 Computer Vision
- **Object Detection:** Identifying and localizing objects in images (e.g., Mask R-CNN)
- **Human Pose Estimation:** Estimating where a person's body joints are in an image/video
- **Semantic Image Editing:** Changing Attributes of a scene (e.g., turning day to night, winter to spring, adding flowers)

### 7.2 Generative Applications
- **Image Synthesis:** Generating realistic fake images (GANs - Generative Adversarial Networks)
- **Style Transfer:** Taking the style of one image (e.g., a Monet painting) and applying it to another (e.g., a photograph)
- **Text-to-Image:** DALL-E and similar models
### 7.3 Vision + Language
- **Image Captioning:** Generating a natural language description of an image(e.g., "A man riding a wave on a surfboard")
- **Visual Question Answering (VQA):** Asking and answering questions about images
- **Video Captioning:** Describing what's happening in a video

### 7.4 Natural Language Processing (NLP)
- **Language Modeling:** Generating coherent text (GPT-3)
- **Machine Translation:** Translating between languages
- **Chatbots:** ChatGPT and similar systems

### 7.5 Games & robotics
- **AlphaGO:** DeepMind's model that beat the world champion at Go (2016). Used reinforcement learning + deep neural networks.
- **AlphaStar:** Played StarCraft II at grandmaster level using multi-agent RL.
- **Rubik's Cube Robot:** OpenAI's robot hand that learned to solve a Rubik's cube (2019). Trained in simulation, transferred to real world.

### 7.6 Healthcare
- **Medical Image Analysis:** Classifying skin cancer from dermoscopy images at dermatologist-level accuracy (Esteva et al., 2017). This is a massive deal  - DL ca potentially screen diseases faster and more accurately than humans.

---

## 8. Transformers & LLMs - The Current Era

The Transformer architecture (Vaswani et al., 2017) is arguably the most important development in modern AI:

- It introduced the concept of **attention mechanisms** - allowing the model to focus on relevant parts of the input.
- It is the backbone of **all modern LLMs**: GPT-3, GPT-4, BERT, CLAUDE, etc.
- It has also been adapted for vision (Vision Transformers), audio, and more.

---
## 9. Tools & Libraries to use

- **NumPy** - Array/Matrix operations (the math backbone)
- **TensorFlow** - DL framework 
- **PyTorch** - DL framework

---
## 10. Key Takeaways

1. **AI > ML > DL** — they are nested, not synonymous. 
2. **ML has multiple paradigms** — supervised, unsupervised, semi-supervised, reinforcement, self-supervised. Supervised learning is the main focus early on. 
3. **DL's superpower** is automatic feature extraction — no manual feature engineering needed. 
4. **DL learns hierarchically** — low-level features → mid-level → high-level → classification. 
5. **Discriminative vs. Generative** — classifying data vs. generating new data. Both powered by DL.
6. **AGI doesn't exist yet.** Everything we build is narrow/specialized AI. 
7. **Transformers (2017)** are the architecture behind the current AI boom.

---

*Completed*

