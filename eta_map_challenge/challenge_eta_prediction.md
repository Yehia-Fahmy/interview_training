# Challenge: Ride ETA Prediction

## Problem Statement

Accurate Estimated Time of Arrival (ETA) predictions are critical for:
- **Driver-passenger matching**: Assigning the right driver to minimize wait times
- **User experience**: Setting correct expectations for riders
- **Surge pricing**: Understanding supply/demand dynamics
- **Route optimization**: Evaluating alternative routes in real-time

Your task is to build a **regression model** that predicts the duration of a ride in minutes, given pickup/dropoff locations, temporal features, and contextual information.

---

## Business Context

- **Goal**: Minimize ETA prediction error to improve rider satisfaction and operational efficiency
- **Constraint**: Model must be fast enough for real-time inference (< 50ms latency)
- **Success Metrics**: 
  - Mean Absolute Error (MAE) < 3 minutes for typical rides
  - Model should handle edge cases (rush hour, bad weather, long distances)
- **Production Consideration**: Model will be called millions of times per day

---

## Dataset Schema

The dataset (`data.csv`) contains historical ride data with the following features:

| Column | Feature Name | Type | Description |
|--------|--------------|------|-------------|
| F1 | pickup_lat | numeric | Pickup latitude coordinate |
| F2 | pickup_lng | numeric | Pickup longitude coordinate |
| F3 | dropoff_lat | numeric | Dropoff latitude coordinate |
| F4 | dropoff_lng | numeric | Dropoff longitude coordinate |
| F5 | distance_miles | numeric | Straight-line (haversine) distance in miles |
| F6 | pickup_datetime | datetime | Timestamp of ride request |
| F7 | day_of_week | numeric | Day of week (0=Monday, 6=Sunday) |
| F8 | hour_of_day | numeric | Hour of pickup (0-23) |
| F9 | is_weekend | binary | 1 if Saturday/Sunday, 0 otherwise |
| F10 | traffic_factor | numeric | Traffic intensity score (1.0=free flow, 2.0+=congested) |
| F11 | weather_condition | categorical | Weather: clear, rain, snow, fog |
| F12 | temperature_f | numeric | Temperature in Fahrenheit |
| F13 | driver_rating | numeric | Driver's average rating (some missing) |
| F14 | vehicle_type | categorical | standard, xl, lux, shared |
| F15 | pickup_zone | categorical | Geographic zone identifier |
| Label | duration_minutes | numeric | Actual ride duration in minutes (TARGET) |

### Data Characteristics

- **Size**: ~12,000 rows
- **Geographic Coverage**: Simulated metropolitan area (similar to Toronto downtown + surrounding areas)
- **Time Range**: Multiple weeks of data across different seasons
- **Missing Values**: Present in `driver_rating`, `temperature_f`, and occasionally other fields
- **Outliers**: Some rides have unusually long durations (traffic incidents, route deviations)
- **Class Distribution**: Continuous target, right-skewed (most rides 5-30 minutes, some 60+)

---

## Requirements

### 1. Data Exploration & Preprocessing (15 minutes)
- Load and explore the dataset
- Handle missing values appropriately (document your strategy)
- Identify and handle outliers
- Feature type validation and conversion

### 2. Feature Engineering (15 minutes)
- Extract useful features from datetime
- Create derived geographic features (e.g., Manhattan distance, bearing)
- Consider interaction features (distance × traffic, hour × is_weekend)
- Handle categorical encoding appropriately

### 3. Model Development (20 minutes)
- Implement at least one regression model
- Split data appropriately (train/validation/test)
- Handle feature scaling if needed
- Consider model complexity vs. inference speed tradeoff

### 4. Evaluation (10 minutes)
- Report appropriate regression metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)
  - R² Score
- Analyze residuals and error distribution
- Identify where the model performs well vs. poorly

### 5. Production Discussion (Final 10-15 minutes with interviewer)
- How would you deploy this model?
- How would you monitor model performance in production?
- What would you do if model performance degrades?
- How would you handle real-time feature computation?

---

## Evaluation Rubric

| Dimension | Weight | What We Look For |
|-----------|--------|------------------|
| Intuition | 25% | Translating ill-defined problem into coherent solution |
| Clarity | 25% | Clean code, good naming, follows best practices |
| Correctness | 25% | Working solution, handles edge cases, reasonable performance |
| Completeness | 15% | Identifies corner cases, discusses limitations |
| Understanding | 10% | Clear explanation of approach and trade-offs |

---

## Stretch Goals

- Implement multiple models and compare (Linear Regression, Random Forest, Gradient Boosting)
- Build feature importance analysis
- Create residual analysis visualization
- Implement cross-validation
- Discuss A/B testing strategy for model rollout
- Consider how to handle cold-start (new drivers, new areas)

---

## Interview Tips

1. **Start by clarifying requirements** - Ask about acceptable error ranges, latency constraints, edge cases
2. **Think out loud** - Explain your reasoning as you make decisions
3. **Prioritize working code** - A simple working solution beats a complex broken one
4. **Manage your time** - Don't spend too long on any single step
5. **Consider production** - Think about how this would work at scale

---

## Getting Started

1. Load the dataset from `data.csv`
2. Explore the data distributions and relationships
3. Implement your preprocessing pipeline
4. Train and evaluate your model
5. Be prepared to discuss your approach and improvements

Good luck!

