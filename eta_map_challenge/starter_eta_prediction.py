"""
Starter Code: Ride ETA Prediction

Your task is to build a regression model that predicts ride duration in minutes.

See challenge_eta_prediction.md for full problem description.

Time Budget:
- Data Exploration & Preprocessing: 15 minutes
- Feature Engineering: 15 minutes  
- Model Development: 20 minutes
- Evaluation: 10 minutes
- Discussion with interviewer: 10-15 minutes

Tips:
- Start simple, iterate if time permits
- Prioritize working code over complex solutions
- Think about production deployment as you code
- Document your assumptions and trade-offs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# You may use any of these libraries (or others you prefer)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =============================================================================
# Data Loading
# =============================================================================

def load_data(path: str = "data.csv") -> pd.DataFrame:
    """Load the dataset from CSV."""
    return pd.read_csv(path)


def explore_data(df: pd.DataFrame) -> None:
    """
    Explore the dataset and print summary statistics.
    
    - Examine target distribution
    - Look for outliers
    - Check feature correlations
    """
    print(f"===== Printing data statistics =====")
    for col in df.columns:
        print(f"col: {col} type: {df[col].dtype} num missing: {df[col].isnull().sum()}")


# =============================================================================
# Preprocessing
# =============================================================================

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    traffic_factor: median temperature -> find avg for hour, day of week, impute with that
    temperature_f: median temperature -> find avg temp for month, impute with that
    driver_rating: give each driver a default value of 2 stars 
    """
    traffic_imputer = SimpleImputer(strategy="median")
    temperature_f_imputer = SimpleImputer(strategy="median")
    driver_imputer = SimpleImputer(strategy="constant", fill_value=2)
    df["traffic_factor"] = traffic_imputer.fit_transform(df[["traffic_factor"]])
    df["temperature_f"] = temperature_f_imputer.fit_transform(df[["temperature_f"]])
    df["driver_rating"] = driver_imputer.fit_transform(df[["driver_rating"]])
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and handle outliers in the dataset.
    
    TODO:
    - Define what constitutes an outlier for this problem
    - Decide whether to remove, cap, or transform outliers
    - Consider business context (some long rides are legitimate)
    """
    return df


def encode_categorical_features(
    df: pd.DataFrame, 
    categorical_cols: list) -> pd.DataFrame:
    """
    Encode categorical features.
    """
    X_cat = df[categorical_cols]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="infrequent_if_exist")
    X_encoded = encoder.fit_transform(X_cat)
    one_hot_cols = encoder.get_feature_names_out(X_cat.columns)
    encoded_df = pd.DataFrame(X_encoded, columns=one_hot_cols, index=df.index)
    df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    return df


def scale_numeric_features(
    df: pd.DataFrame,
    numeric_cols: list) -> pd.DataFrame:
    """
    Scale numeric features.
    """
    X_num = df[numeric_cols]
    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X_num)
    X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=df.index)
    df[numeric_cols] = X_scaled_df
    return df


# =============================================================================
# Feature Engineering
# =============================================================================

def create_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from datetime column.
    """
    df = df.drop(["pickup_datetime"], axis=1)
    return df


def create_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived geographic features.
    """
    coord_cols = ["pickup_lat", "pickup_lng", "dropoff_lat", "dropoff_lng"]
    df = df.drop(coord_cols, axis=1)
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features.
    
    TODO:
    - distance × traffic_factor
    - hour × is_weekend
    - weather × distance
    - Other meaningful interactions
    """
    pass


# =============================================================================
# Model Training
# =============================================================================

def split_data(
    df: pd.DataFrame, 
    target_col: str = "duration_minutes",
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[list, list, list, list, list, list]:
    """
    Split data into train, validation, and test sets.
    
    TODO:
    - Consider stratification if needed
    - Ensure temporal ordering if relevant
    - Return X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop([target_col], axis=1)
    Y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size+val_size, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_size/(test_size+val_size), shuffle=True)
    return (X_train, X_test, X_val, y_train, y_test, y_val)

class CustomDataset(Dataset):
    def __init__(self, features: pd.DataFrame, labels: pd.Series) -> None:
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.float32).view(-1,1)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index], self.labels[index]

class TimePredictionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(22)
        self.linear1 = nn.Linear(22, 50)

        self.bn2 = nn.BatchNorm1d(50)
        self.linear2 = nn.Linear(50, 100)

        self.bn3 = nn.BatchNorm1d(100)
        self.linear3 = nn.Linear(100, 25)

        self.bn4 = nn.BatchNorm1d(25)
        self.linear4 = nn.Linear(25, 1)

    def forward(self, x):
        x = self.relu(self.linear1(self.bn1(x)))

        x = self.relu(self.linear2(self.bn2(x)))

        x = self.relu(self.linear3(self.bn3(x)))

        x = self.relu(self.linear4(self.bn4(x)))

        return x
        

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    Train a regression model.
    """
    training_dataset = CustomDataset(X_train, y_train)
    training_loader = DataLoader(training_dataset, 64, True, num_workers=4)
    model = TimePredictionModel()
    opt = Adam(model.parameters())
    loss_function = nn.L1Loss()
    model = model.to(torch.device("mps"))
    model.train()
    for _ in range(10):
        cumulative_loss = 0
        for inputs, labels in training_loader:
            inputs, labels = inputs.to(torch.device("mps")), labels.to(torch.device("mps"))
            opt.zero_grad()
            out = model(inputs)
            loss = loss_function(out, labels)
            cumulative_loss += loss.item()
            loss.backward()
            opt.step()
        print(f"cumulative_loss = {cumulative_loss}, avg_loss = {cumulative_loss/len(training_loader)}")
    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    model: Any, 
    X_test: pd.DataFrame, 
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    TODO:
    - Calculate MAE, RMSE, MAPE, R²
    - Analyze residuals
    - Identify where model performs well/poorly
    - Return metrics dictionary
    """
    testing_dataset = CustomDataset(X_test, y_test)
    testing_loader = DataLoader(testing_dataset, 64, True, num_workers=4)
    model = model.to(torch.device("mps"))
    loss_function = nn.L1Loss()
    model.eval()
    cumulative_loss = 0
    for inputs, labels in testing_loader:
        inputs, labels = inputs.to(torch.device("mps")), labels.to(torch.device("mps"))
        out = model(inputs)
        loss = loss_function(out, labels)
        cumulative_loss += loss.item()
    print(f"cumulative_loss = {cumulative_loss}, avg_loss = {cumulative_loss/len(testing_loader)}")

def analyze_errors(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    original_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Analyze prediction errors.
    
    TODO:
    - Plot residual distribution
    - Identify systematic errors
    - Find worst predictions and understand why
    """
    pass


# =============================================================================
# Pipeline
# =============================================================================

class ETAPredictionPipeline:
    """
    Complete pipeline for ETA prediction.
    
    This class should encapsulate the entire workflow from raw data
    to predictions, making it easy to deploy in production.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_columns = None
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "ETAPredictionPipeline":
        """
        Fit the entire pipeline on training data.
        
        TODO:
        - Preprocess data
        - Engineer features
        - Train model
        - Store all transformers for inference
        """
        pass
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        TODO:
        - Apply same preprocessing as training
        - Handle missing features gracefully
        - Return predictions in minutes
        """
        pass
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate pipeline on test data.
        
        TODO:
        - Make predictions
        - Calculate metrics
        - Return results
        """
        pass


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function to run the ETA prediction pipeline."""
    
    # Set data path
    data_path = Path(__file__).parent / "data.csv"
    
    # TODO: Implement your solution here
    # 
    # Suggested workflow:
    # 1. Load and explore data
    # 2. Preprocess (handle missing values, outliers)
    # 3. Engineer features
    # 4. Split data
    # 5. Train model
    # 6. Evaluate and analyze results
    # 7. Be ready to discuss production considerations
    
    print("=" * 60)
    print("Lyft ETA Prediction Challenge")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    df = load_data(data_path)
    
    # Explore data
    print("\n[2] Exploring data...")
    explore_data(df)
    
    # Preprocess
    print("\n[3] Preprocessing...")
    df = handle_missing_values(df)
    df = handle_outliers(df)
    categorical_features = ["weather_condition", "vehicle_type", "pickup_zone"]
    numeric_features = ["pickup_lat", "pickup_lng", "dropoff_lat", "dropoff_lng", "distance_miles", "day_of_week", "hour_of_day", "traffic_factor", "temperature_f", "driver_rating"]
    df = encode_categorical_features(df, categorical_features)
    df = scale_numeric_features(df, numeric_features)
    
    # Feature engineering
    print("\n[4] Engineering features...")
    df = create_datetime_features(df)
    df = create_geographic_features(df)
    # df = create_interaction_features(df)
    
    # Split data
    print("\n[5] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    
    # Train model
    print("\n[6] Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate
    print("\n[7] Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

