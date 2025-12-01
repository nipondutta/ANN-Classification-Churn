# ANN-Classification-Churn
Customer Churn Prediction

# 📊 Customer Churn Prediction using ANN

This project builds an **Artificial Neural Network (ANN)** model to predict **customer churn** — whether a customer is likely to leave a service.  
The model is trained on structured customer data and uses deep learning for binary classification.

---

## 🚀 Project Overview

Customer churn is a critical business metric. Using machine learning, we can identify customers who are at risk of leaving and take preventive action.

### Key Objectives
- Perform exploratory data analysis (EDA)
- Preprocess categorical and numeric features
- Build an ANN using TensorFlow/Keras
- Evaluate and predict churn probability

---

## 🧠 Model Architecture

| Layer | Type | Units | Activation |
|------|------|-------|------------|
| Input | Dense | depends on features | ReLU |
| Hidden Layer 1 | Dense | 6 | ReLU |
| Hidden Layer 2 | Dense | 6 | ReLU |
| Output | Dense | 1 | Sigmoid |

Optimizer: **Adam**  
Loss Function: **Binary Crossentropy**  
Evaluation Metric: **Accuracy**

---
