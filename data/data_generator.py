import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_samples=1000, random_seed=42):
    np.random.seed(random_seed)
    
    # 70% healthy (0), 30% not healthy (1)
    labels = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
    
    data = []
    
    for label in labels:
        # Base independent features
        age_years = round(np.random.uniform(0.1, 15.0), 1)
        breed_size = np.random.choice(["small", "medium", "large"])
        
        # Weight depends slightly on breed
        if breed_size == "small":
            weight_kg = round(np.random.uniform(2.0, 10.0), 1)
        elif breed_size == "medium":
            weight_kg = round(np.random.uniform(10.0, 30.0), 1)
        else:
            weight_kg = round(np.random.uniform(30.0, 90.0), 1)
            
        vaccination_status = np.random.choice([0, 1], p=[0.2, 0.8])
        
        # Adjust features based on health status (label)
        if label == 0:  # Healthy
            body_temp_c = round(np.random.normal(38.7, 0.3), 1)
            # bound
            body_temp_c = max(37.5, min(39.5, body_temp_c))
            
            heart_rate_bpm = int(np.random.normal(100, 15))
            heart_rate_bpm = max(60, min(140, heart_rate_bpm))
            
            activity_level = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
            appetite_level = np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])
            
            # Healthy dogs are mostly vaccinated
            if np.random.rand() < 0.9:
                vaccination_status = 1
                
        else:  # Not Healthy
            # Could have fever or low temp, wide variation
            temp_type = np.random.choice(["fever", "low", "normal"], p=[0.6, 0.2, 0.2])
            if temp_type == "fever":
                body_temp_c = round(np.random.uniform(39.5, 41.5), 1)
            elif temp_type == "low":
                body_temp_c = round(np.random.uniform(37.5, 38.0), 1)
            else:
                body_temp_c = round(np.random.uniform(38.0, 39.5), 1)
                
            hr_type = np.random.choice(["high", "low", "normal"], p=[0.5, 0.3, 0.2])
            if hr_type == "high":
                heart_rate_bpm = int(np.random.uniform(140, 180))
            elif hr_type == "low":
                heart_rate_bpm = int(np.random.uniform(60, 80))
            else:
                heart_rate_bpm = int(np.random.uniform(80, 140))
                
            activity_level = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            appetite_level = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            
            # Unvaccinated dogs have a slightly higher chance of being unhealthy
            if np.random.rand() < 0.4:
                vaccination_status = 0
                
        # Clip strictly to requested bounds just in case
        body_temp_c = max(37.5, min(41.5, body_temp_c))
        heart_rate_bpm = max(60, min(180, heart_rate_bpm))
        
        row = {
            "age_years": age_years,
            "weight_kg": weight_kg,
            "body_temp_c": body_temp_c,
            "heart_rate_bpm": heart_rate_bpm,
            "vaccination_status": vaccination_status,
            "activity_level": activity_level,
            "appetite_level": appetite_level,
            "breed_size": breed_size,
            "label": label
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Optional: shuffle dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic canine health dataset...")
    df = generate_synthetic_data(1000)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, "raw_data.csv")
    df.to_csv(out_file, index=False)
    
    print(f"Dataset generated with shape {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts(normalize=True)}")
    print(f"Saved to {out_file}")
