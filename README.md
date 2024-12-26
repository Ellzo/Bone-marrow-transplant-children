# Bone Marrow Transplant in Children

This repository contains the code and report for analyzing factors affecting survival rates in children undergoing hematopoietic stem cell transplantation (HSCT). The project leverages machine learning techniques to identify key predictors of survival and evaluate various models for predictive accuracy.

## Repository Contents

- `Main.ipynb`: A Jupyter notebook containing all the code for data preprocessing, feature selection, visualization, model training, hyperparameter optimization, and evaluation.
- `Report.pdf`: A detailed report documenting the methodology, results, and analysis of the project.

## Project Overview

Hematopoietic stem cell transplantation (HSCT) is a critical medical procedure for treating severe blood disorders. This project analyzes a dataset of HSCT cases to:

- Identify the factors influencing survival rates.
- Build predictive models using machine learning.
- Provide insights to guide patient selection and management.

### Key Features
- Comprehensive feature selection using domain knowledge, correlation analysis, and model-based importance.
- Evaluation of multiple machine learning models, including Decision Trees, Random Forests, K-Nearest Neighbors, Naive Bayes, and Artificial Neural Networks.
- Hyperparameter optimization using Grid Search and Random Search.
- Addressing challenges such as data leakage and imbalanced datasets.

## Dataset

The dataset consists of 149 entries with 37 attributes covering:
- Donor characteristics (e.g., age, blood group, CMV status).
- Recipient characteristics (e.g., age, gender, disease type).
- Compatibility metrics (e.g., HLA match, ABO compatibility).
- Post-transplant outcomes (e.g., survival status, relapse, GvHD).

## Methodology

The analysis involved:
1. **Data Preprocessing**: Cleaning and organizing data for analysis.
2. **Feature Selection**: Selecting predictive features using statistical and model-based methods.
3. **Model Training**: Evaluating various classifiers with cross-validation.
4. **Hyperparameter Optimization**: Fine-tuning models for better performance.
5. **Model Evaluation**: Comparing models using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.

## Results and Findings

- Random Forest and Artificial Neural Networks emerged as the best-performing models.
- Significant features included donor-recipient compatibility metrics, disease type, and CMV status.
- Addressing data leakage ensured realistic model performance.

## Installation and Requirements

To run the code in `Main.ipynb`, install the following dependencies:

```plaintext
pandas
numpy
scikit-learn
matplotlib
seaborn
torch
```

Install the dependencies with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch
```

## Usage
1. Open `Main.ipynb` in Jupyter Notebook or Jupyter Lab.
2. Run the cells sequentially to reproduce the analysis.

## License
This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
You are free to share and adapt the material under the following terms:
* **Attribution:** You must give appropriate credit, provide a link to the license, and indicate if changes were made.
* **NonCommercial:** You may not use the material for commercial purposes.

For more details, see the LICENSE file or visit https://creativecommons.org/licenses/by-nc/4.0/.
