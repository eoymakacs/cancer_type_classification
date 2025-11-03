# 🧬 Cancer Type Classification from Gene Expression

## Overview
This project predicts **cancer type** using **gene expression profiles**.  
It is a multiclass classification problem where each sample contains expression levels for thousands of genes.  

The model is implemented using **Python** and **TensorFlow Keras**.

---

## Dataset
The dataset is **not included** in this repository due to its large size (~196 MB).  
You need **two CSV files**:

1. `data.csv` – gene expression values (samples × genes)  
2. `labels.csv` – sample IDs and corresponding cancer type labels  

**Sample of `labels.csv`:**

| Sample ID   | Class |
|------------|-------|
| sample_0   | PRAD  |
| sample_1   | LUAD  |
| sample_2   | PRAD  |

**Download options:**

- Kaggle: [Pan-Cancer Gene Expression Dataset](https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq)  
- Alternatively, other TCGA gene expression datasets can be used.

**Place both CSV files in the `data/` folder** in the root directory.

---

## Project Structure
```plaintext
root/
├── src/
│ └── model_training.py # Script for preprocessing, training, and evaluation
├── README.md
└── data/ # Place downloaded CSVs here (not tracked by Git)
├── data.csv
└── labels.csv
```

---

## Preprocessing
1. Load `data.csv` and `labels.csv`  
2. Align samples by their IDs  
3. Scale gene expression values using **StandardScaler**  
4. Encode cancer type labels into **one-hot vectors** for neural network training  

---

## Neural Network Architecture

- Input layer: number of genes (~20,531)  
- Dense layer 1: 512 units, ReLU activation  
- Dropout: 0.3  
- Dense layer 2: 256 units, ReLU activation  
- Dropout: 0.2  
- Dense layer 3: 128 units, ReLU activation  
- Output layer: number of cancer types, softmax activation  

- **Loss:** categorical crossentropy  
- **Optimizer:** Adam  
- **Metrics:** Accuracy  

---

## Training

```bash
cd src
python model_training.py
```

- Epochs: 50
- Batch size: 32
- Validation split: 20%

---

## Model Evaluation

- Test Accuracy
- Confusion Matrix (per cancer type)
- Classification Report (Precision, Recall, F1-score)

Example:
  ```python
  from sklearn.metrics import classification_report, confusion_matrix
  import seaborn as sns
  import matplotlib.pyplot as plt
  ```
- Plot confusion matrix and analyze misclassifications
- Visualize training/validation accuracy over epochs

---

## 🚀 Next Steps

- **Dimensionality Reduction:**  
  Apply **PCA (Principal Component Analysis)** or **autoencoders** to reduce input dimensionality and improve training speed.

- **Model Comparison:**  
  Experiment with alternative algorithms like **Random Forest**, **XGBoost**, or **Support Vector Machines (SVM)** to compare performance with the neural network.

- **Hyperparameter Tuning:**  
  Use **GridSearchCV** or **KerasTuner** to optimize model parameters such as learning rate, dropout rate, and layer sizes.

- **Cross-Validation:**  
  Implement **k-fold cross-validation** to ensure that results are consistent and not dependent on a single train-test split.

- **Model Explainability:**  
  Use interpretability tools like **SHAP** or **LIME** to identify which genes have the most influence on specific cancer predictions.

- **Deployment:**  
  Conver

---

## 📚 References

- [The Cancer Genome Atlas (TCGA) Program](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)  
  Large-scale dataset providing molecular profiles for over 30 cancer types.

- [Kaggle: Gene Expression Cancer RNA-Seq Data](https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq)  
  Example dataset for multiclass cancer classification tasks.

- [TensorFlow Keras Documentation](https://www.tensorflow.org/guide/keras)  
  Official guide for building and training deep learning models with TensorFlow.

- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)  
  Reference for preprocessing, feature scaling, and evaluation metrics.

- [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/)  
  Library for model explainability and feature importance visualization.

- [LIME (Local Interpretable Model-agnostic Explanations)](https://github.com/marcotcr/lime)  
  Framework for interpreting predictions of machine learning models.


