"""
Data Generation Script: Ride ETA Prediction Dataset

This script generates synthetic ride data for the Lyft Laptop Interview challenge.
The data is designed to be realistic with:
- Geographic patterns (city center vs suburbs)
- Temporal patterns (rush hour, weekends)
- Weather effects on ride duration
- Missing values and outliers (real-world messiness)

Run this script to generate data.csv for the challenge.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# Configuration
# =============================================================================

# Toronto-like metropolitan area bounds
CITY_CENTER = (43.6532, -79.3832)  # Downtown Toronto coordinates
CITY_BOUNDS = {
    'lat_min': 43.58,
    'lat_max': 43.85,
    'lng_min': -79.65,
    'lng_max': -79.20
}

# Zone definitions (simplified geographic zones)
ZONES = [
    {'name': 'downtown', 'center': (43.6532, -79.3832), 'radius': 0.03, 'traffic_mult': 1.5},
    {'name': 'midtown', 'center': (43.6800, -79.4000), 'radius': 0.04, 'traffic_mult': 1.3},
    {'name': 'east_end', 'center': (43.6700, -79.3200), 'radius': 0.05, 'traffic_mult': 1.1},
    {'name': 'west_end', 'center': (43.6500, -79.4500), 'radius': 0.05, 'traffic_mult': 1.2},
    {'name': 'north', 'center': (43.7500, -79.4000), 'radius': 0.06, 'traffic_mult': 1.0},
    {'name': 'airport', 'center': (43.6777, -79.6248), 'radius': 0.02, 'traffic_mult': 1.4},
    {'name': 'suburbs', 'center': (43.7800, -79.3500), 'radius': 0.08, 'traffic_mult': 0.9},
]

VEHICLE_TYPES = ['standard', 'xl', 'lux', 'shared']
VEHICLE_WEIGHTS = [0.60, 0.20, 0.10, 0.10]
VEHICLE_SPEED_MULT = {'standard': 1.0, 'xl': 0.95, 'lux': 1.0, 'shared': 0.85}

WEATHER_CONDITIONS = ['clear', 'rain', 'snow', 'fog']
WEATHER_WEIGHTS = [0.65, 0.20, 0.10, 0.05]
WEATHER_SPEED_MULT = {'clear': 1.0, 'rain': 0.85, 'snow': 0.70, 'fog': 0.80}

# Dataset size
NUM_RIDES = 12000


# =============================================================================
# Helper Functions
# =============================================================================

def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate haversine distance in miles between two points."""
    R = 3959  # Earth's radius in miles
    
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def get_zone(lat: float, lng: float) -> str:
    """Determine which zone a coordinate falls into."""
    for zone in ZONES:
        center_lat, center_lng = zone['center']
        dist = haversine_distance(lat, lng, center_lat, center_lng)
        if dist < zone['radius'] * 69:  # Convert degrees to approximate miles
            return zone['name']
    return 'suburbs'


def get_zone_traffic_multiplier(zone_name: str) -> float:
    """Get traffic multiplier for a zone."""
    for zone in ZONES:
        if zone['name'] == zone_name:
            return zone['traffic_mult']
    return 1.0


def generate_coordinates(n: int, bias_downtown: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic pickup/dropoff coordinates.
    
    Biases toward downtown for more realistic distribution.
    """
    lats = []
    lngs = []
    
    for _ in range(n):
        if np.random.random() < bias_downtown:
            # Bias toward downtown/midtown
            zone = np.random.choice(ZONES[:4], p=[0.4, 0.3, 0.15, 0.15])
            center_lat, center_lng = zone['center']
            # Add noise around zone center
            lat = center_lat + np.random.normal(0, 0.015)
            lng = center_lng + np.random.normal(0, 0.02)
        else:
            # Random location within city bounds
            lat = np.random.uniform(CITY_BOUNDS['lat_min'], CITY_BOUNDS['lat_max'])
            lng = np.random.uniform(CITY_BOUNDS['lng_min'], CITY_BOUNDS['lng_max'])
        
        # Clip to bounds
        lat = np.clip(lat, CITY_BOUNDS['lat_min'], CITY_BOUNDS['lat_max'])
        lng = np.clip(lng, CITY_BOUNDS['lng_min'], CITY_BOUNDS['lng_max'])
        
        lats.append(lat)
        lngs.append(lng)
    
    return np.array(lats), np.array(lngs)


def generate_datetime(n: int, start_date: str = "2024-01-01") -> pd.DatetimeIndex:
    """Generate realistic pickup datetimes with temporal patterns."""
    start = pd.to_datetime(start_date)
    end = start + timedelta(days=90)  # 3 months of data
    
    datetimes = []
    
    for _ in range(n):
        # Random day
        day_offset = np.random.randint(0, 90)
        base_date = start + timedelta(days=day_offset)
        
        # Hour with realistic distribution (more rides during commute/evening)
        hour_probs = np.array([
            0.01, 0.005, 0.005, 0.005, 0.01, 0.02,  # 0-5 AM
            0.04, 0.08, 0.10, 0.08, 0.06, 0.06,     # 6-11 AM
            0.07, 0.06, 0.05, 0.05, 0.06, 0.08,     # 12-5 PM
            0.10, 0.08, 0.06, 0.05, 0.04, 0.02      # 6-11 PM
        ])
        hour_probs = hour_probs / hour_probs.sum()
        hour = np.random.choice(24, p=hour_probs)
        
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        dt = base_date.replace(hour=hour, minute=minute, second=second)
        datetimes.append(dt)
    
    return pd.DatetimeIndex(datetimes)


def calculate_traffic_factor(hour: int, day_of_week: int, is_weekend: bool, 
                            pickup_zone: str, weather: str) -> float:
    """
    Calculate traffic factor based on time, location, and weather.
    
    Returns multiplier where 1.0 = free flow, 2.0+ = heavy congestion.
    """
    # Base traffic by hour
    if 7 <= hour <= 9:  # Morning rush
        base_traffic = 1.6
    elif 16 <= hour <= 19:  # Evening rush
        base_traffic = 1.7
    elif 11 <= hour <= 14:  # Lunch time
        base_traffic = 1.2
    elif 22 <= hour or hour <= 5:  # Night
        base_traffic = 0.9
    else:
        base_traffic = 1.1
    
    # Weekend reduction
    if is_weekend:
        base_traffic *= 0.7
    
    # Zone multiplier
    zone_mult = get_zone_traffic_multiplier(pickup_zone)
    
    # Weather effect
    weather_mult = WEATHER_SPEED_MULT.get(weather, 1.0)
    # Inverse for traffic (bad weather = more traffic)
    weather_traffic = 1 / weather_mult
    
    # Combine factors with some randomness
    traffic = base_traffic * zone_mult * weather_traffic
    traffic *= np.random.uniform(0.85, 1.15)  # Random variation
    
    return max(0.8, min(2.5, traffic))  # Clip to reasonable range


def calculate_duration(distance: float, traffic_factor: float, weather: str,
                       vehicle_type: str, is_shared: bool = False) -> float:
    """
    Calculate realistic ride duration in minutes.
    
    Base formula: duration = (distance / avg_speed) * 60 + pickup_time
    With adjustments for traffic, weather, vehicle type, etc.
    """
    # Base average speed in mph (city driving)
    base_speed = 18  # Typical city average speed
    
    # Adjust for traffic
    effective_speed = base_speed / traffic_factor
    
    # Weather adjustment
    weather_mult = WEATHER_SPEED_MULT.get(weather, 1.0)
    effective_speed *= weather_mult
    
    # Vehicle type adjustment
    vehicle_mult = VEHICLE_SPEED_MULT.get(vehicle_type, 1.0)
    effective_speed *= vehicle_mult
    
    # Calculate base duration
    if effective_speed > 0:
        travel_time = (distance / effective_speed) * 60  # Convert to minutes
    else:
        travel_time = distance * 5  # Fallback
    
    # Add pickup/navigation time (1-5 minutes)
    pickup_time = np.random.uniform(1, 4)
    
    # Add random delays (traffic lights, stops, etc.)
    random_delay = np.random.exponential(1.5)
    
    # Shared rides take longer (multiple stops)
    if vehicle_type == 'shared':
        shared_delay = np.random.uniform(3, 10)
    else:
        shared_delay = 0
    
    total_duration = travel_time + pickup_time + random_delay + shared_delay
    
    # Add occasional outliers (accidents, road closures, wrong routes)
    if np.random.random() < 0.03:  # 3% chance of major delay
        total_duration *= np.random.uniform(1.5, 2.5)
    
    # Minimum duration is 3 minutes (even for very short trips)
    return max(3.0, total_duration)


def generate_weather(n: int, datetimes: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray]:
    """Generate weather conditions correlated with season/time."""
    conditions = []
    temperatures = []
    
    for dt in datetimes:
        month = dt.month
        hour = dt.hour
        
        # Seasonal base temperature (Fahrenheit)
        if month in [12, 1, 2]:  # Winter
            base_temp = 25
            snow_prob = 0.25
            rain_prob = 0.10
        elif month in [3, 4, 5]:  # Spring
            base_temp = 50
            snow_prob = 0.05
            rain_prob = 0.25
        elif month in [6, 7, 8]:  # Summer
            base_temp = 75
            snow_prob = 0.0
            rain_prob = 0.15
        else:  # Fall
            base_temp = 50
            snow_prob = 0.08
            rain_prob = 0.20
        
        # Time of day adjustment
        if 6 <= hour <= 10:
            temp_adj = -5
        elif 14 <= hour <= 18:
            temp_adj = 5
        else:
            temp_adj = 0
        
        # Random variation
        temp = base_temp + temp_adj + np.random.normal(0, 8)
        temperatures.append(temp)
        
        # Weather condition
        rand = np.random.random()
        if rand < snow_prob and temp < 35:
            conditions.append('snow')
        elif rand < snow_prob + rain_prob:
            conditions.append('rain')
        elif rand < snow_prob + rain_prob + 0.05:
            conditions.append('fog')
        else:
            conditions.append('clear')
    
    return np.array(conditions), np.array(temperatures)


def introduce_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Introduce realistic missing values into the dataset."""
    df = df.copy()
    
    # driver_rating: 8% missing (new drivers, system errors)
    mask = np.random.random(len(df)) < 0.08
    df.loc[mask, 'driver_rating'] = np.nan
    
    # temperature_f: 5% missing (sensor errors)
    mask = np.random.random(len(df)) < 0.05
    df.loc[mask, 'temperature_f'] = np.nan
    
    # traffic_factor: 3% missing (API failures)
    mask = np.random.random(len(df)) < 0.03
    df.loc[mask, 'traffic_factor'] = np.nan
    
    # Some edge cases with multiple missing
    mask = np.random.random(len(df)) < 0.01
    df.loc[mask, ['driver_rating', 'temperature_f']] = np.nan
    
    return df


# =============================================================================
# Main Data Generation
# =============================================================================

def generate_dataset(n: int = NUM_RIDES) -> pd.DataFrame:
    """Generate the complete ride dataset."""
    print(f"Generating {n} rides...")
    
    # Generate pickup coordinates
    pickup_lat, pickup_lng = generate_coordinates(n, bias_downtown=0.45)
    
    # Generate dropoff coordinates (with some correlation to pickup)
    dropoff_lat, dropoff_lng = generate_coordinates(n, bias_downtown=0.35)
    
    # Calculate distances
    distances = np.array([
        haversine_distance(plat, plng, dlat, dlng)
        for plat, plng, dlat, dlng in zip(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
    ])
    
    # Filter out very short rides (< 0.3 miles) and regenerate
    short_ride_mask = distances < 0.3
    while short_ride_mask.any():
        n_short = short_ride_mask.sum()
        new_dlat, new_dlng = generate_coordinates(n_short, bias_downtown=0.3)
        dropoff_lat[short_ride_mask] = new_dlat
        dropoff_lng[short_ride_mask] = new_dlng
        distances[short_ride_mask] = [
            haversine_distance(plat, plng, dlat, dlng)
            for plat, plng, dlat, dlng in zip(
                pickup_lat[short_ride_mask], pickup_lng[short_ride_mask],
                new_dlat, new_dlng
            )
        ]
        short_ride_mask = distances < 0.3
    
    # Cap very long rides
    distances = np.clip(distances, 0.3, 25)
    
    # Generate datetimes
    datetimes = generate_datetime(n)
    
    # Extract temporal features
    day_of_week = datetimes.dayofweek.values
    hour_of_day = datetimes.hour.values
    is_weekend = (day_of_week >= 5).astype(int)
    
    # Generate weather
    weather_conditions, temperatures = generate_weather(n, datetimes)
    
    # Determine zones
    pickup_zones = np.array([get_zone(lat, lng) for lat, lng in zip(pickup_lat, pickup_lng)])
    
    # Generate vehicle types
    vehicle_types = np.random.choice(VEHICLE_TYPES, size=n, p=VEHICLE_WEIGHTS)
    
    # Generate driver ratings (4.0 - 5.0, right skewed)
    driver_ratings = 5.0 - np.random.exponential(0.3, size=n)
    driver_ratings = np.clip(driver_ratings, 4.0, 5.0)
    driver_ratings = np.round(driver_ratings, 2)
    
    # Calculate traffic factors
    traffic_factors = np.array([
        calculate_traffic_factor(hour, dow, weekend, zone, weather)
        for hour, dow, weekend, zone, weather in zip(
            hour_of_day, day_of_week, is_weekend, pickup_zones, weather_conditions
        )
    ])
    
    # Calculate durations
    durations = np.array([
        calculate_duration(dist, traffic, weather, vehicle)
        for dist, traffic, weather, vehicle in zip(
            distances, traffic_factors, weather_conditions, vehicle_types
        )
    ])
    
    # Round durations to 1 decimal
    durations = np.round(durations, 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'pickup_lat': np.round(pickup_lat, 6),
        'pickup_lng': np.round(pickup_lng, 6),
        'dropoff_lat': np.round(dropoff_lat, 6),
        'dropoff_lng': np.round(dropoff_lng, 6),
        'distance_miles': np.round(distances, 2),
        'pickup_datetime': datetimes,
        'day_of_week': day_of_week,
        'hour_of_day': hour_of_day,
        'is_weekend': is_weekend,
        'traffic_factor': np.round(traffic_factors, 2),
        'weather_condition': weather_conditions,
        'temperature_f': np.round(temperatures, 1),
        'driver_rating': driver_ratings,
        'vehicle_type': vehicle_types,
        'pickup_zone': pickup_zones,
        'duration_minutes': durations
    })
    
    # Introduce missing values
    df = introduce_missing_values(df)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def print_dataset_stats(df: pd.DataFrame) -> None:
    """Print summary statistics of the generated dataset."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    print(f"\nShape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    for col, count in missing[missing > 0].items():
        print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nTarget variable (duration_minutes):")
    print(f"  Min: {df['duration_minutes'].min():.1f}")
    print(f"  Max: {df['duration_minutes'].max():.1f}")
    print(f"  Mean: {df['duration_minutes'].mean():.1f}")
    print(f"  Median: {df['duration_minutes'].median():.1f}")
    print(f"  Std: {df['duration_minutes'].std():.1f}")
    
    print(f"\nDistance (miles):")
    print(f"  Min: {df['distance_miles'].min():.2f}")
    print(f"  Max: {df['distance_miles'].max():.2f}")
    print(f"  Mean: {df['distance_miles'].mean():.2f}")
    
    print(f"\nCategorical distributions:")
    print(f"\nWeather conditions:\n{df['weather_condition'].value_counts()}")
    print(f"\nVehicle types:\n{df['vehicle_type'].value_counts()}")
    print(f"\nPickup zones:\n{df['pickup_zone'].value_counts()}")
    
    print(f"\nTemporal distribution:")
    print(f"  Weekday rides: {(df['is_weekend'] == 0).sum()}")
    print(f"  Weekend rides: {(df['is_weekend'] == 1).sum()}")
    
    print("\n" + "=" * 60)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    # Generate dataset
    df = generate_dataset(NUM_RIDES)
    
    # Print statistics
    print_dataset_stats(df)
    
    # Save to CSV
    output_path = Path(__file__).parent / "data.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

