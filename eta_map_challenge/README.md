# Lyft Laptop Interview Challenge: ETA Prediction

## Overview

You will build a regression model to predict ride duration (ETA) given various features about the ride.

## Challenge Structure

| File | Description |
|------|-------------|
| `challenge_eta_prediction.md` | Full problem description, requirements, and evaluation criteria |
| `starter_eta_prediction.py` | Minimal starter code with placeholder functions |
| `data.csv` | Dataset with ~12,000 historical rides |
| `generate_eta_data.py` | Data generation script (for reference only) |

## Quick Start

1. **Read the challenge description**:
   ```bash
   cat challenge_eta_prediction.md
   ```

2. **Generate the dataset** (if not already present):
   ```bash
   python generate_eta_data.py
   ```

3. **Start coding**:
   ```bash
   # Open starter_eta_prediction.py and implement your solution
   python starter_eta_prediction.py
   ```

## Time Breakdown (90 minutes total)

| Phase | Time | Focus |
|-------|------|-------|
| Problem Review | 15 min | Read problem, explore data, ask clarifying questions |
| Coding | 60 min | Implement preprocessing, feature engineering, modeling |
| Discussion | 15 min | Discuss approach, trade-offs, production considerations |

## What We Look For

- **Intuition**: Can you translate an ill-defined problem into a working solution?
- **Clarity**: Is your code clean, well-organized, and easy to understand?
- **Correctness**: Does your solution work and handle edge cases?
- **Completeness**: Do you identify limitations and corner cases?
- **Understanding**: Can you explain your decisions and trade-offs?

## Interview Tips

1. **Clarify requirements first** - Don't assume, ask questions
2. **Start simple** - A working simple solution beats a broken complex one
3. **Think out loud** - Explain your reasoning as you code
4. **Manage time** - Don't spend too long on any single step
5. **Consider production** - Think about deployment, monitoring, scalability

## Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| `pickup_lat/lng` | numeric | Pickup coordinates |
| `dropoff_lat/lng` | numeric | Dropoff coordinates |
| `distance_miles` | numeric | Straight-line distance |
| `pickup_datetime` | datetime | Timestamp of ride |
| `day_of_week` | numeric | 0=Monday, 6=Sunday |
| `hour_of_day` | numeric | 0-23 |
| `is_weekend` | binary | Weekend indicator |
| `traffic_factor` | numeric | Traffic intensity (1.0=free flow) |
| `weather_condition` | categorical | clear, rain, snow, fog |
| `temperature_f` | numeric | Temperature in Fahrenheit |
| `driver_rating` | numeric | Driver's average rating |
| `vehicle_type` | categorical | standard, xl, lux, shared |
| `pickup_zone` | categorical | Geographic zone |
| **`duration_minutes`** | numeric | **TARGET: Ride duration** |

## Data Characteristics

- ~12,000 rows
- Missing values in `driver_rating`, `temperature_f`, `traffic_factor`
- Some outliers (unusually long rides)
- Temporal patterns (rush hour, weekends)
- Geographic patterns (downtown vs suburbs)

## Evaluation Metrics

- **MAE** (Mean Absolute Error) - Primary metric
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of determination)

## Production Considerations to Discuss

- How would you deploy this model?
- How would you handle real-time feature computation?
- How would you monitor model performance?
- What would you do if performance degrades?
- How would you handle new pickup zones or vehicle types?

---

Good luck! Remember: we want to see how you think and approach problems, not just the final answer.

