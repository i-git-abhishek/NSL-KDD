# Network Intrusion Detection: Baseline ML vs. Graph Attention Networks

## Project Overview
The objective of this project is to evaluate the efficacy of standard machine learning algorithms against custom Graph Neural Network (GNN) architectures for network intrusion detection. 

Modern network traffic datasets often saturate standard classification metrics when processed by traditional distance-based or ensemble models. This project establishes rigorous baselines using algorithms like Multi-Layer Perceptrons (MLP) and Random Forests, and compares them against a custom Graph Attention Network (GAT) built in PyTorch. The GAT specifically investigates whether mapping tabular network data into a topological graph (via K-Nearest Neighbors) improves the recall of minority attack vectors by analyzing structural node relationships rather than isolated features.

## Dataset
This project utilizes the **NSL-KDD** dataset, a refined version of the KDD'99 dataset standard in cybersecurity research. 
* **Data Splits Evaluated:** Both the standard 20% subset (for rapid prototyping and hyperparameter tuning) and the complete 125,000+ record dataset.
* Note: The raw dataset files are excluded from this repository via `.gitignore` due to size constraints.

## Methodology

### 1. Data Preprocessing
* Applied standard scaling to all continuous features to ensure mathematical stability for distance-based baselines (e.g., SVM).
* Encoded categorical network features (protocols, services, flags).
* Applied class weighting techniques across models to penalize false negatives (missed attacks) due to the inherent class imbalance between Normal traffic and specific Attack categories.

### 2. Baseline Models
* **Support Vector Machine (SVM):** Configured with an RBF kernel and scaled inputs.
* **Random Forest Classifier:** Tuned for maximum depth and class-weight balancing.
* **Multi-Layer Perceptron (MLP):** A 3-hidden-layer deep neural network (128, 64, 32) utilized as the primary high-performance baseline.

### 3. Graph Attention Network (GAT) Architecture
* **Graph Construction:** Converted tabular network data into a transductive graph format using a K-Nearest Neighbors (K-NN) algorithm based on Euclidean distance in the feature space.
* **Network Design:** Implemented using PyTorch Geometric, utilizing Multi-Head Attention to allow the model to learn multiple independent representations of the network topology.

## Key Results (MLP Baseline)
The Multi-Layer Perceptron baseline achieved near-perfect classification on the full dataset split. 

* **Overall Accuracy:** 99.65%
* **Testing Sample Size:** 18,895

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Normal** | 1.00 | 1.00 | 1.00 | 10,145 |
| **Attack** | 1.00 | 1.00 | 1.00 | 8,750 |

**Confusion Matrix Summary:**
* True Negatives (Normal correctly flagged): 10,115
* False Positives (Normal flagged as Attack): 30
* False Negatives (Attack missed): 36
* True Positives (Attack correctly flagged): 8,714

*(See `mlp_classifier_matrix.png` in the repository for the visual matrix).*

**Research Note:** The saturation of metrics (>99% accuracy) is a known characteristic when applying deep learning to the standard NSL-KDD dataset. Future experimentation will involve aggressive feature starvation and artificially limiting training data volume to stress-test the robustness of the GAT architecture against the MLP baseline under zero-day constraints.

## Repository Structure
```text
NSL-KDD/
│
├── exploration and experimentaion/   # Jupyter notebooks containing individual model tests, EDA, and hyperparameter tuning
├── results/                          # Output metrics, graphs, and logs from the 20% dataset subset
├── results_complete_dataset/         # Output metrics, graphs, and logs from the full 125k dataset
│
├── nsl-kdd/                          # Raw dataset directory (Excluded from version control)
│
├── Unified_all_models.ipynb          # Main execution notebook containing the end-to-end pipeline for all baselines and GAT
├── mlp_classifier_matrix.png         # Exported confusion matrix for the baseline MLP
└── Readme.md                         # Project documentation