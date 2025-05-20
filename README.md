# NYC Taxi Trip Duration Prediction

## 📊 About The Project

This project explores machine learning (ML) and deep learning (DL) approaches to predict taxi trip durations in New York City using Apache PySpark. By analyzing key trip attributes such as pickup and drop-off locations, weather conditions, and time-based features, we aim to enhance efficiency for both passengers and drivers in one of the world's busiest urban transportation networks.

## 🔍 Key Findings

- Traditional ML models, particularly gradient boosting, outperformed deep learning models for this structured tabular data
- Gradient boosting achieved the highest accuracy (70.38%) among ML models
- LSTM networks were the best performing deep learning approach with 71.43% accuracy
- Clustering revealed meaningful patterns in taxi trip data across New York City

## 📋 Dataset Features

The NYC Taxi Trip dataset includes:
- Trip ID: A unique identifier for each ride
- Pickup and Drop-off Times: Timestamps for trip start and end
- Pickup and Drop-off Locations: Latitude and longitude coordinates
- Trip Distance: Distance traveled during the ride
- Fare Information: Total fare amount, tips, and surcharges
- Passenger Count: Number of passengers per ride

## 🛠️ Methods & Models

### Data Preprocessing
- Handling missing and invalid values
- Feature engineering (hour of day, day of week, weather conditions)
- Normalization and scaling
- Categorical variable encoding

### Machine Learning Models
- Linear Regression (baseline model)
- Gradient Boosted Trees Regression
- K-Means Clustering

### Deep Learning Models
- Convolutional Neural Networks (CNN)
- Long Short-Term Memory (LSTM) networks

### Evaluation Metrics
- R² Score
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Accuracy Within 10% Range

## 📈 Results

| Model | RMSE | Accuracy (%) |
|-------|------|-------------|
| Linear Regression | 0.504 | 50.39 |
| Gradient Boosted Trees | 367.352 | 70.38 |
| CNN | - | 64.29 |
| LSTM | - | 71.43 |

## 📊 Visualizations

The project includes several key visualizations:
- Trip and passenger count by weekday
- Mean distance and trip duration by weekday
- Distance and trip duration correlation
- K-means clustering with feature importance
- Radar plots showing feature distribution across clusters

## 🔬 Insights

- Highest taxi demand occurs on Fridays and Saturdays
- Trip durations tend to be longer on weekdays due to traffic congestion
- Strong positive correlation exists between trip distance and duration
- Certain clusters show distinct patterns based on time of day and location

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Apache Spark 3.1.2+
- PySpark
- PyTorch/TensorFlow
- Scikit-Learn
- NumPy
- Pandas
- Matplotlib

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nyc-taxi-trip-prediction.git
cd nyc-taxi-trip-prediction

# Set up virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Run data preprocessing
python src/preprocessing.py

# Train and evaluate models
python src/train_models.py

# Generate visualizations
python src/visualize.py
```

## 🔮 Future Work

- Feature engineering enhancement with external data sources
- Implementation of hybrid ML-DL models
- Real-time prediction system integration
- Incorporation of traffic data and road conditions
- Deployment of a web dashboard for trip duration estimation

## 👥 Contributors

- [Hasin Md. Daiyan]
- [Rebeka Sultana]
- [Shahriar Hossain]
- [Ekramul Huda Chowdhury]

## 📚 References

1. Zhao, X., Wang, S., & Ye, Z. (2020). "Geospatial Analysis of Urban Mobility Patterns Using Big Data." Journal of Urban Studies.
2. Jiang, Y., He, Y., & Li, Z. (2018). "Clustering High-Demand Taxi Zones: A Case Study on New York City." Transportation Research Record.
3. Shekhar, S., Xiong, H., & Zhou, C. (2019). "Spatial Big Data Challenges and Opportunities." Springer.
4. Gößling, S., Cohen, S., & Hares, A. (2019). "Temporal Trends in Urban Taxi Use." Urban Transport Journal.
5. Wang, J., Lin, H., & Zhang, T. (2021). "Anomaly Detection in Urban Mobility Patterns Using PySpark." IEEE Transactions on Big Data.


