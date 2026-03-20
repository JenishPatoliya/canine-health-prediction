import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix
import shap
import joblib
import warnings

warnings.filterwarnings('ignore')

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv"))["label"]
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv"))["label"]
    return X_train, X_test, y_train, y_test, base_dir

def train_and_evaluate():
    X_train, X_test, y_train, y_test, base_dir = load_data()
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    metrics = ["accuracy", "roc_auc", "precision", "recall", "f1"]
    
    results = {}
    print("Evaluating models with 5-fold CV...")
    for name, model in models.items():
        # Evaluate with 5-fold CV
        cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=metrics)
        results[name] = {metric: np.mean(cv_results[f"test_{metric}"]) for metric in metrics}
        print(f"--- {name} ---")
        for metric in metrics:
            print(f"{metric}: {results[name][metric]:.4f}")
            
    plots_dir = os.path.join(base_dir, "app", "assets")
    os.makedirs(plots_dir, exist_ok=True)
            
    # Train all models on train data to get ROC curves on test set
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
        
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "roc_curves.png"))
    plt.close()
    
    # Select best model based on CV ROC-AUC
    best_model_name = max(results, key=lambda x: results[x]["roc_auc"])
    print(f"\nBest model based on CV ROC-AUC: {best_model_name}")
    
    # Tune best model with RandomizedSearchCV
    best_model = models[best_model_name]
    
    print("Tuning best model...")
    if best_model_name == "Random Forest":
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    elif best_model_name == "XGBoost":
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    else:
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l2']
        }
        
    random_search = RandomizedSearchCV(
        best_model, param_distributions=param_grid, n_iter=10, 
        scoring='roc_auc', cv=5, random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    
    final_model = random_search.best_estimator_
    print(f"Best params: {random_search.best_params_}")
    
    # Final evaluation on test set
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:, 1]
    
    print("\n--- Final Model Test Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1: {f1_score(y_test, y_pred):.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({best_model_name})')
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()
    
    # Save the model
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(final_model, os.path.join(models_dir, "best_model.pkl"))
    print("Best model saved.")
    
    # SHAP plots
    print("Generating SHAP plots...")
    
    if best_model_name in ["Random Forest", "XGBoost"]:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test)
        
        # Output shape diff for RF vs XGB depending on shap version
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        explainer = shap.LinearExplainer(final_model, X_train)
        shap_values = explainer.shap_values(X_test)
        
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(plots_dir, "shap_summary.png"), bbox_inches='tight')
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(os.path.join(plots_dir, "shap_bar.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    train_and_evaluate()
