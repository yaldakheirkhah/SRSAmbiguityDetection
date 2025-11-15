# Ambiguity Detection in Software Requirements Using BERT

This repository contains the official implementation of my MSc research project titled  
**"BERT Empowered: A Transformer Approach to Ambiguity Recognition in Software Requirements Documents."**

The project explores how transformer-based language models, specifically **BERT**, can identify and categorize ambiguities in software requirements. Ambiguous requirements are a major cause of miscommunication, delays, and costly errors in software development, and this work proposes an intelligent NLP-based solution to mitigate these issues.

---

## 🚀 Project Overview

Traditional rule-based or classical ML models struggle to capture the contextual subtleties of natural language. This project leverages **BERT**, a transformer-based model, to:

- Detect ambiguous sentences in requirements documents  
- Classify different types of ambiguity  
- Reduce false positives and false negatives  
- Provide suggestions to users for clarification  
- Improve iteratively through reinforcement learning

The model achieved an **F1-score of 0.9259**, showing strong performance in balancing precision and recall.

---

## 🧠 Key Features

- **Transformer-based ambiguity detection**
- **Fine-tuned BERT model**
- **Custom dataset of software requirements**
- **Multi-class ambiguity classification**
- **Reinforcement learning loop for continuous improvement**
- **Jupyter notebook with full pipeline**

---

## 📂 Repository Structure

```
/SRSAmbiguityDetection
│── rf-transformer.ipynb        # Main notebook with full implementation
│── final.csv                   # Dataset used for training/testing
│── requirements.txt                  
│── README.md
```

---

## 📊 Dataset

The dataset contains software requirement sentences labeled for ambiguity classes.

Example format:

| text                                   | label        |
|----------------------------------------|--------------|
| The system should load quickly.        | vague_term   |
| The user may optionally enable X.      | optionality  |

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/yaldakheirkhah/SRSAmbiguityDetection.git
cd SRSAmbiguityDetection
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Open and run the notebook:

```
jupyter notebook rf-transformer.ipynb
```

The notebook includes preprocessing, model training, evaluation, and analysis.

---

## 👩‍💻 Author

**Yalda Kheirkhah**  
MSc in Computer Engineering – NLP Researcher  
Email: **yalda.kheirkhah@iau.ir**

This project was conducted as part of my Master’s thesis at Islamic Azad University.

---

## 📄 License  
This project is released for academic review purposes.
