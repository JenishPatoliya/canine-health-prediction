import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

def preprocess_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw_data.csv")
    
    # 1. Load raw_data.csv
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # 2. Encode breed_size with LabelEncoder
    le = LabelEncoder()
    df["breed_size"] = le.fit_transform(df["breed_size"])
    
    # Also save the LabelEncoder to use during inference in the app
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))
    
    X = df.drop("label", axis=1)
    y = df["label"]
    
    # 3. Train/test split 80/20 with stratify=y
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 4. Build a Pipeline to avoid data leakage (using imblearn pipeline to include SMOTE)
    print("Building pipeline with SMOTE and StandardScaler...")
    # We define the scaler separately to save it later easily, or extract from pipeline
    scaler = StandardScaler()
    smote = SMOTE(random_state=42)
    
    pipeline = ImbPipeline([
        ('scaler', scaler),
        ('smote', smote)
    ])
    
    # 5. Handle class imbalance using SMOTE and Scale using the pipeline
    # fit_resample is applied to training data only to avoid data leakage
    print("Applying pipeline to training data...")
    X_train_res_np, y_train_res_np = pipeline.fit_resample(X_train, y_train)
    
    # The test set is only transformed via the scaler, SMOTE is not applied to test data
    X_test_scaled_np = pipeline.named_steps['scaler'].transform(X_test)
    
    # Convert back to dataframes
    feature_cols = X.columns
    X_train_res = pd.DataFrame(X_train_res_np, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=feature_cols)
    y_train_res = pd.Series(y_train_res_np, name="label")
    y_test = y_test.reset_index(drop=True)
    
    # Save the processed datasets for model training
    print("Saving processed datasets...")
    data_dir = os.path.join(base_dir, "data")
    X_train_res.to_csv(os.path.join(data_dir, "X_train.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(data_dir, "X_test.csv"), index=False)
    y_train_res.to_csv(os.path.join(data_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(data_dir, "y_test.csv"), index=False)
    
    # 6. Save scaler
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(pipeline.named_steps['scaler'], scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_data()
