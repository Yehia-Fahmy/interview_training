"""
Dataset Generator for App Deletion Risk Prediction Challenge

Generates a realistic dataset with missing values for the ML interview challenge.
Run this script to create data.csv in the same directory.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SAMPLES = 15000
POSITIVE_CLASS_RATIO = 0.20  # ~20% deletion rate
OUTPUT_PATH = Path(__file__).parent / "data.csv"

# Feature distributions
COUNTRIES = ["US", "UK", "BR", "IN", "AR"]
COUNTRY_PROBS = [0.35, 0.20, 0.15, 0.20, 0.10]

GENDERS = ["M", "F", "U"]
GENDER_PROBS = [0.45, 0.45, 0.10]

DOWNLOAD_CHANNELS = ["E", "P", "O", "PA"]  # Email, Push, Organic, Paid
CHANNEL_PROBS = [0.15, 0.25, 0.40, 0.20]

PLATFORMS = ["I", "A"]  # iOS, Android
PLATFORM_PROBS = [0.45, 0.55]


def generate_base_features(n: int) -> pd.DataFrame:
    """Generate base features without label dependency."""
    
    # Country (F1)
    country = np.random.choice(COUNTRIES, size=n, p=COUNTRY_PROBS)
    
    # Gender (F2)
    gender = np.random.choice(GENDERS, size=n, p=GENDER_PROBS)
    
    # Age (F3) - normal distribution centered around 32
    age = np.clip(np.random.normal(32, 12, n), 17, 87).astype(int)
    
    # Download channel (F4)
    download_channel = np.random.choice(DOWNLOAD_CHANNELS, size=n, p=CHANNEL_PROBS)
    
    # Platform (F12)
    platform = np.random.choice(PLATFORMS, size=n, p=PLATFORM_PROBS)
    
    # Time of day (F10) - bimodal: morning and evening peaks
    time_of_day = np.zeros(n, dtype=int)
    morning_mask = np.random.random(n) < 0.4
    time_of_day[morning_mask] = np.clip(np.random.normal(9, 2, morning_mask.sum()), 0, 23).astype(int)
    time_of_day[~morning_mask] = np.clip(np.random.normal(20, 3, (~morning_mask).sum()), 0, 23).astype(int)
    
    # Day of week (F11) - slightly higher on weekends
    day_weights = [0.12, 0.12, 0.12, 0.12, 0.14, 0.19, 0.19]
    day_of_week = np.random.choice(7, size=n, p=day_weights)
    
    # App version (F13) - most users on recent versions
    app_version = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], size=n, 
                                    p=[0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.15, 0.12, 0.08])
    
    # OS version (F14)
    os_version = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=n,
                                   p=[0.02, 0.03, 0.05, 0.10, 0.15, 0.25, 0.25, 0.15])
    
    return pd.DataFrame({
        "F1": country,
        "F2": gender,
        "F3": age,
        "F4": download_channel,
        "F10": time_of_day,
        "F11": day_of_week,
        "F12": platform,
        "F13": app_version,
        "F14": os_version,
    })


def generate_behavioral_features(n: int, base_df: pd.DataFrame) -> pd.DataFrame:
    """Generate behavioral features that will correlate with deletion probability."""
    
    # Base engagement score (hidden variable that drives correlations)
    engagement_score = np.random.beta(2, 5, n)  # Skewed towards lower engagement
    
    # Boost engagement for organic users
    organic_mask = base_df["F4"] == "O"
    engagement_score[organic_mask] *= 1.3
    engagement_score = np.clip(engagement_score, 0, 1)
    
    # Interactions 7d (F5) - highly correlated with engagement
    interactions_7d = (engagement_score * 50 + np.random.exponential(3, n)).astype(int)
    interactions_7d = np.clip(interactions_7d, 0, 100)
    
    # Interactions 14d (F6) - should be >= 7d interactions
    interactions_14d = interactions_7d + (engagement_score * 30 + np.random.exponential(5, n)).astype(int)
    interactions_14d = np.clip(interactions_14d, interactions_7d, 150)
    
    # Interactions 30d (F7) - should be >= 14d interactions
    interactions_30d = interactions_14d + (engagement_score * 40 + np.random.exponential(8, n)).astype(int)
    interactions_30d = np.clip(interactions_30d, interactions_14d, 200)
    
    # Daily time spent (F8) - in minutes, correlated with engagement
    daily_time_spent = (engagement_score * 45 + np.random.exponential(5, n)).astype(int)
    daily_time_spent = np.clip(daily_time_spent, 1, 60)
    
    # Time since last use (F9) - inversely correlated with engagement
    time_since_last_use = ((1 - engagement_score) * 15 + np.random.exponential(2, n)).astype(int)
    time_since_last_use = np.clip(time_since_last_use, 1, 30)
    
    return pd.DataFrame({
        "F5": interactions_7d,
        "F6": interactions_14d,
        "F7": interactions_30d,
        "F8": daily_time_spent,
        "F9": time_since_last_use,
    }), engagement_score


def generate_labels(n: int, base_df: pd.DataFrame, behavioral_df: pd.DataFrame, 
                    engagement_score: np.ndarray) -> np.ndarray:
    """Generate labels based on features to create realistic correlations."""
    
    # Base deletion probability from engagement score (inverse relationship)
    deletion_prob = 1 - engagement_score
    deletion_prob = deletion_prob * 0.4  # Scale down
    
    # Increase deletion probability for users with high time_since_last_use
    high_inactivity = behavioral_df["F9"] > 10
    deletion_prob[high_inactivity] += 0.15
    
    # Decrease deletion probability for iOS users (typically higher retention)
    ios_mask = base_df["F12"] == "I"
    deletion_prob[ios_mask] *= 0.85
    
    # Paid users have slightly lower deletion (invested)
    paid_mask = base_df["F4"] == "PA"
    deletion_prob[paid_mask] *= 0.9
    
    # Older app versions have higher deletion
    old_version_mask = base_df["F13"] < 5
    deletion_prob[old_version_mask] += 0.1
    
    # Clip probabilities
    deletion_prob = np.clip(deletion_prob, 0.05, 0.7)
    
    # Adjust to hit target positive class ratio
    current_mean = deletion_prob.mean()
    adjustment = POSITIVE_CLASS_RATIO / current_mean
    deletion_prob = np.clip(deletion_prob * adjustment, 0.01, 0.95)
    
    # Generate labels from probabilities
    labels = (np.random.random(n) < deletion_prob).astype(int)
    
    return labels


def introduce_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Introduce realistic missing value patterns."""
    
    df = df.copy()
    n = len(df)
    
    # Gender (F2) - ~10% missing
    missing_mask = np.random.random(n) < 0.10
    df.loc[missing_mask, "F2"] = ""
    
    # Age (F3) - ~8% missing
    missing_mask = np.random.random(n) < 0.08
    df.loc[missing_mask, "F3"] = np.nan
    
    # Download channel (F4) - ~12% missing (not always tracked)
    missing_mask = np.random.random(n) < 0.12
    df.loc[missing_mask, "F4"] = ""
    
    # Daily time spent (F8) - ~5% missing (tracking issues)
    missing_mask = np.random.random(n) < 0.05
    df.loc[missing_mask, "F8"] = np.nan
    
    # Time since last use (F9) - ~3% missing
    missing_mask = np.random.random(n) < 0.03
    df.loc[missing_mask, "F9"] = np.nan
    
    # Platform (F12) - ~2% missing
    missing_mask = np.random.random(n) < 0.02
    df.loc[missing_mask, "F12"] = ""
    
    # App version (F13) - ~4% missing
    missing_mask = np.random.random(n) < 0.04
    df.loc[missing_mask, "F13"] = np.nan
    
    # OS version (F14) - ~6% missing
    missing_mask = np.random.random(n) < 0.06
    df.loc[missing_mask, "F14"] = np.nan
    
    return df


def generate_dataset() -> pd.DataFrame:
    """Generate the complete dataset."""
    
    print(f"Generating {N_SAMPLES} samples...")
    
    # Generate base features
    base_df = generate_base_features(N_SAMPLES)
    
    # Generate behavioral features
    behavioral_df, engagement_score = generate_behavioral_features(N_SAMPLES, base_df)
    
    # Combine features
    df = pd.concat([base_df[["F1", "F2", "F3", "F4"]], 
                    behavioral_df,
                    base_df[["F10", "F11", "F12", "F13", "F14"]]], axis=1)
    
    # Generate labels
    labels = generate_labels(N_SAMPLES, base_df, behavioral_df, engagement_score)
    df["Label"] = labels
    
    # Introduce missing values
    df = introduce_missing_values(df)
    
    # Reorder columns to match schema
    column_order = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", 
                    "F10", "F11", "F12", "F13", "F14", "Label"]
    df = df[column_order]
    
    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of the generated dataset."""
    
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    
    print(f"\nShape: {df.shape}")
    print(f"\nClass distribution:")
    print(df["Label"].value_counts(normalize=True))
    
    print(f"\nMissing values per column:")
    missing = df.isnull().sum() + (df == "").sum()
    missing_pct = (missing / len(df) * 100).round(2)
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing_pct[col]}%)")
    
    print(f"\nSample rows:")
    print(df.head(10).to_string())


if __name__ == "__main__":
    # Generate dataset
    df = generate_dataset()
    
    # Print summary
    print_dataset_summary(df)
    
    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset saved to: {OUTPUT_PATH}")

