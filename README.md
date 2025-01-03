# Basic_ML_Projects

Welcome to the **Basic_ML_Projects** repository! This repository contains a collection of beginner to intermediate-level machine learning projects aimed at solving real-world problems across diverse domains. Each project is designed to enhance your understanding of machine learning concepts, data preprocessing, model development, and deployment.

---

## 📁 Repository Structure

Each project in this repository has its own directory, containing:

1. **Project Code**: Python code for data preprocessing, model training, evaluation, and predictions.
2. **Dataset**: The dataset used for the project (or instructions to access it if it's too large to include).
3. **Deployment Code**: Code to deploy the trained model using frameworks like Flask, FastAPI, or Streamlit.

---

## 📚 List of Projects

### 1. **Heart Disease Prediction**
- **Objective**: Predict the likelihood of heart disease in patients based on medical attributes.
- **Tech Stack**: LightGBM Classifier, GridSearchCV for hyperparameter tuning.
- **Key Metrics**: Accuracy: 95.83% (Testing).

### 2. **Liver Damage Stage Prediction**
- **Objective**: Predict the stage of liver damage in patients using categorical and numerical medical data.
- **Tech Stack**: Stratified train-test split, Label Encoding, LightGBM Classifier.
- **Key Metrics**: MSE: 0.017 (Testing).

### 3. **Lung Cancer Survivability Prediction**
- **Objective**: Predict the survivability of lung cancer patients using a large and imbalanced dataset.
- **Tech Stack**: SMOTE for class balancing, XGBoost Classifier.
- **Key Metrics**: High accuracy and performance evaluation.

### 4. **Mobile Price Prediction**
- **Objective**: Predict the price range of mobile phones (e.g., cheap, moderate, high) based on technical specifications.
- **Tech Stack**: XGBoost, GridSearchCV for tuning.
- **Key Metrics**: MSE: 0.0124 (Training), Accuracy: 0.0618 (Testing).

### 5. **Thyroid Cancer Recurrence Prediction**
- **Objective**: Predict the recurrence of thyroid cancer in patients.
- **Tech Stack**: Support Vector Classifier (SVC), GridSearchCV for tuning.
- **Key Metrics**: Accuracy: 94.88% (Testing).

### 6. **Titanic Survivability Prediction**
- **Objective**: Predict the survivability of passengers on the Titanic based on demographic and socioeconomic features.
- **Tech Stack**: Logistic Regression, SVC, GridSearchCV.
- **Key Metrics**: Best accuracy achieved with tuned SVC.

### 7. **Turnover Prediction by Stock Data**
- **Objective**: Predict company turnover based on stock data using regression analysis.
- **Tech Stack**: Linear Regression.
- **Key Metrics**: MSE: 0.03.

### 8. **Vehicle Price Prediction**
- **Objective**: Predict vehicle prices based on features such as brand, type, and technical specifications.
- **Tech Stack**: LightGBM Regressor, StandardScaler for target scaling.
- **Key Metrics**: MSE: 0.065 (Training), 0.2 (Testing).

---

## 💻 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Basic_ML_Projects.git
   ```

2. Navigate to the desired project directory:
   ```bash
   cd Basic_ML_Projects/<project_name>
   ```

3. Follow the instructions in the respective project directory to run the project code, explore the dataset, and deploy the model.

---

## 🚀 Deployment
Each project includes deployment code to make the trained model accessible via an API or web interface. Instructions for deploying the model can be found in the respective project directories.

---

## 🤝 Contributions
Contributions are welcome! If you want to add a new project, improve existing ones, or fix issues, please fork the repository, create a branch, and submit a pull request.

---

### 🌟 Happy Learning and Building!
