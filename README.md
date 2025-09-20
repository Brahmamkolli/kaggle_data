## Titanic Survival Prediction (Kaggle)

This project focuses on predicting passenger survival on the Titanic dataset from Kaggle using various machine learning models. The notebooks include data preprocessing, feature engineering, model training, hyperparameter tuning, and submission file generation for Kaggle.

📂 Repository Structure

Kaggle Train Data.ipynb
Covers business problem understanding, data preprocessing, balancing the dataset with SMOTE, feature scaling, and training multiple ML models:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Decision Tree

Random Forest

AdaBoost

Gradient Boosting

XGBoost

Final trained model is exported using joblib.

Kaggle_Test_data.ipynb
Handles test dataset preprocessing (consistent with training), including:

Applying saved Box-Cox transformation parameters

Handling missing values

Preparing features for prediction with the final model

Kaggle Random Forest Data.ipynb
Focused on Random Forest and additional pipelines:

Data splitting with StratifiedShuffleSplit

Custom preprocessing pipelines (imputation, encoding, feature dropping, scaling)

GridSearchCV hyperparameter tuning for RandomForest & Linear SVC

Submission file creation (KAGGLEsubmission.xlsx)

⚙️ Installation & Requirements

Clone the repo and install dependencies:

git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt

Requirements

Python 3.x

pandas, numpy, matplotlib, seaborn

scikit-learn

imbalanced-learn (for SMOTE)

xgboost

joblib

🚀 How to Run

Open Kaggle Train Data.ipynb → Train models and save the best one.

Open Kaggle_Test_data.ipynb → Apply preprocessing and run predictions.

Open Kaggle Random Forest Data.ipynb → Experiment with Random Forest pipeline and generate Kaggle submission file.

📊 Results

Models were evaluated using accuracy and cross-validation scores.

Feature selection and transformations (Box-Cox, scaling, SMOTE) improved generalization.

Final predictions are stored in an .xlsx file for Kaggle submission.

🔮 Future Improvements

Try ensemble stacking of top models.

Perform deeper feature engineering (family size, titles, cabin grouping).

Use SHAP/feature importance for explainability.
