# Air pollution prediction platform (Sarajevo)

Collaborators: Andrijana Kešelj, Anđela Maksimović and Šehzada Sijarić

This project is done by the students of DSAI at ETF Sarajevo.

## Table of Contents
- [Overview](#overview)
- [Problem: Air Pollution in Sarajevo](#problem-air-pollution-in-sarajevo)
- [Our Objective](#our-objective)
- [Dataset Choice](#dataset-choice)
- [Data Preparation & EDA](#data-preparation--eda)
- [Our Initial Approach](#our-initial-approach)
- [Initial Results and Evaluation](#initial-results-and-evaluation)
- [Project Challenges](#project-challenges)
- [Improvements](#improvements)
- [Final Results](#final-results)
- [Performance Visualization](#performance-visualization)
- [Air Quality Dashboard](#air-quality-dashboard)
- [Conclusion and Future Steps](#conclusion-and-future-steps)
- [References](#references)
- [License](#license)

## **Overview**
This project focuses on predicting air pollution levels in Sarajevo at the neighborhood level using machine learning models. To capture different characteristics of the data, we applied a GRU-based time-series model for PM10 prediction and an XGBoost model for PM2.5 prediction. The overall goal is to provide reliable short-term predictions (7-days) that can support air quality monitoring, and are able to inform citizens about AQI and its harmful health-related consequences.
## **Problem: Air pollution in Sarajevo**
Air pollution has been a long-standing issue in Sarajevo, with serious consequences for public health and the environment. The problem is driven by factors such as industrial emissions, heavy traffic, and the city’s geographical location. During winter months, temperature inversions trap pollutants near the ground, often turning Sarajevo into a smog-filled city and significantly worsening air quality.

As a result, Sarajevo is frequently ranked among the most polluted cities in the world, with other cities across Bosnia and Herzegovina facing similar challenges. Prolonged exposure to polluted air poses major health risks, including respiratory and cardiovascular diseases, particularly for vulnerable groups such as children, the elderly, and individuals with pre-existing health conditions.
## **Our objective**
The goal of this project is to develop an ML-powered air quality prediction platform capable of forecasting PM2.5 and PM10 levels at the neighborhood level in Sarajevo, with potential future expansion to other cities.

The platform aims to provide personal exposure alerts and health-focused recommendations, helping residents take precautionary measures and supporting actions to improve air quality.


## **Dataset choice**
We used publicly available air quality data from reference monitoring stations located across multiple neighborhoods in Sarajevo (e.g., Otoka, Bjelave, U.S. Embassy, Ilijaš, etc..)

The raw dataset can be found on this [link](https://aqicn.org/map/sarajevo/)
We concatenated the datasets of all stations into one large dataset.
Different station datasets start at different times, but all data end at a common cutoff date of November 19, 2025.
## **Data preparation & EDA**
The dataset underwent a rigorous cleaning pipeline to ensure data quality and suitability for modeling.

Standardization: Converted column names to lowercase and stripped whitespace. Non-standard missing value placeholders (e.g., "NA", "null", "--") were identified and converted to numerical NaN.
Feature Removal: The CO (Carbon Monoxide) column was dropped entirely. Analysis revealed >40% missing data; imputing such a large volume would have introduced significant noise and bias.
Feature Engineering: Derived temporal features (Year, Month, Season) from the raw timestamp to enable seasonal analysis.

**Final dataset after cleaning consists of:**
Total records: 19164
Date range: 2016-11-07 00:00:00 to 2025-11-19 00:00:00
Stations: 7

**EDA observations:**
Highest PM2.5: US Embassy station.
Highest PM10: Ilijas station.
Cleanest: Ivan Sedlo (mountain area) consistently has the best air.
When Nitrogen Dioxide (NO2) and Sulfur Dioxide (SO2) levels rise, Particulate Matter (PM) usually rises too. Ozone (O3) usually drops when PM gets high.

<img width="749" height="590" alt="image" src="https://github.com/user-attachments/assets/42683ba6-ce37-4d58-a7b1-f0afe262a90d" />

PM2.5​ vs. PM10​: The correlation between the two particle sizes is a moderate 0.50, confirming that while they are related (as PM2.5​ is a subset of PM10​), they are distinct phenomena, with approximately half of the PM10​ variability being explained by PM2.5​ variability.

<img width="1489" height="590" alt="image" src="https://github.com/user-attachments/assets/ec8d85b2-d47a-4de3-a27a-213c1457662e" />

PM2.5 is consistent across the region, while PM10 varies more by location. NO2 and SO2 are useful predictors because they share combustion sources. Despite sharp winter peaks, overall pollution has gradually declined since 2016

<img width="902" height="590" alt="image" src="https://github.com/user-attachments/assets/cbbc8710-9a8e-4825-b319-dc62395fabad" />

The heatmap shows that PM2.5 concentrations are consistently high across all locations, confirming it is a widespread regional issue, while PM10 varies significantly, peaking in the industrial area of Ilijas and dropping at Ivan Sedlo. Notably, the PM2.5 values are consistently higher than PM10.

## **Our initial approach**
In the initial stage of the project, we explored a wide range of modeling techniques for air quality prediction. We started with simple regression models and Random Forest as baselines, then moved on to more advanced machine learning methods such as XGBoost, Support Vector Regression, and other bagging and boosting approaches.

To better capture temporal patterns, we also experimented with deep learning models, including LSTM and GRU architectures, using different lookback windows (7-day and 14-day), single-step and multi-step forecasting, global and per-station models, and seasonal configurations. Both single-output and multi-output models were tested to compare pollutant-specific training against joint prediction. Additionally, we evaluated different grouping strategies, including:

Per-station models: Training separate models for each monitoring station to capture local patterns.

Global models: Training a single model on combined data from all stations to improve generalization.

Seasonal grouping: Training models for specific seasons (e.g., winter vs summer) to better capture seasonal variation in pollutant behavior.

These experiments helped us identify suitable models for different pollutants and forecasting horizons.
## **Initial results and evaluation**  
The initial results showed clear differences in performance across models and pollutants. Tree-based models, especially XGBoost, performed best for PM2.5, while GRU models achieved stronger results for PM10. When combining these two models, prediction accuracy (measured by MAE) was highest for the first forecast day and gradually decreased from days 2 to 7, which is expected for time-series forecasting.
## **Project challenges**
During model training, we noticed inconsistent results, which led us to review the air quality data sources. We found that the Vijećnica station uses PM2.5 and O₃ data from the Sarajevo Bjelave station, while the U.S. Embassy relies on PM10, NO₂, SO₂, CO, and O₃ measurements from the same location. In addition, the Ilidža station uses CO and O₃ data from the Otoka station. 
This explains the inconsistencies observed during training.
Another challenge appeared nearing the end of the project: We  added European air quality standards to the user interface. However, these standards are relatively strict, which means the system rarely reports good AQI values and often classifies the air quality as poor. This explains why the displayed results frequently indicate bad air quality, even when pollutant levels are not extreme.
## **Improvements**
Building on our initial approach, we implemented several key improvements to enhance model accuracy, robustness, and interpretability:

1. **Hybrid Modeling Strategy**  
   - **PM2.5**: we used XGBoost with 7-day lookback for high interpretability and robustness to outliers  
   - **PM10**: implemented a 2-layer GRU with 30-day lookback for capturing long-term temporal dependencies  
   - This leverages each algorithm's strengths for different pollutant characteristics

2. **Advanced Feature Engineering**  
   - Created time-lagged features (lag1, lag7) and rolling statistics for temporal patterns  
   - Added cyclical date encoding (sine/cosine of day-of-year) to capture seasonal variations  
   - Engineered pollutant interaction features (like O₃ inverse relationship)

3. **Log-Residual Transformation for PM10**  
   - Applied log1p transformation to stabilize variance and reduce outlier impact  
   - Predicted residuals relative to last observed value rather than absolute concentrations  
   - This improved model stability and reduced error propagation in multi-day forecasts

4. **Robust Scaling and Regularization**  
   - Used RobustScaler for PM10 features to handle extreme values  
   - Implemented dropout layers (0.4→0.3) and L2 regularization in GRU architecture  
   - Added early stopping and learning rate reduction callbacks

5. **Comprehensive Validation Framework**  
   - Implemented time-aware splits (70/15/15) respecting temporal order  
   - Created custom RMSE monitoring callback with model checkpointing  
   - Generated detailed audit reports with skill scores and hit rates

6. **Enhanced Visualization Dashboard**  
   - Developed 12-panel dashboard comparing model performance across stations  
   - Added forecast horizon analysis showing accuracy decay over 7 days  
   - Included seasonal pattern visualization and feature importance analysis

## **Final results**
**PM2.5 (XGBoost) Performance**
- **Overall Average R²**: 0.853
- **Best Station**: Ilidza (R² = 0.961, RMSE = 9.47)
- **Worst Station**: Ivan Sedlo (R² = 0.691, RMSE = 24.99)
- **Average RMSE**: 18.4 μg/m³
- **Average MAE**: 12.4 μg/m³

**Performance Categories for PM2.5:**
- **Excellent (R² > 0.9)**: Ilidza, Otoka
- **Good (0.8 < R² ≤ 0.9)**: Bjelave, US Embassy, Vijecnica
- **Moderate (R² ≤ 0.8)**: Ilijas, Ivan Sedlo

 **PM10 (GRU) Performance**
- **Overall Average R²**: 0.452
- **Best Station**: Ilidza (R² = 0.629, RMSE = 13.45)
- **Worst Station**: Vijecnica (R² = 0.163, RMSE = 39.98)
- **Average RMSE**: 19.7 μg/m³
- **Average MAE**: 9.8 μg/m³
- **7-Day Forecast Horizon**: Accuracy decreases by ~2.1 RMSE/day on average

## **Performance Visualization**

<img width="1255" height="586" alt="download (36)" src="https://github.com/user-attachments/assets/944c4faf-4fc0-462c-a990-536d0804a3b3" />

This plot shows that the predicted values closely follow the actual measurements throughout the observed period, showing very good model performance. The model captures both short-term fluctuations and larger seasonal trends, with only small deviations during sharp peaks and sudden drops. Higher concentrations toward the later months are well reflected in the predictions, suggesting that the model is able to learn temporal patterns effectively. Overall, the high R² value confirms that the model explains most of the variability in the data and provides reliable forecasts for this station

## **Air Quality Dashboard**
We developed a web-based air quality dashboard using Streamlit, combining an XGBoost model for PM2.5 and a GRU model for PM10 into a single interactive interface. The application allows users to explore historical data, generate forecasts, and interpret air quality using a Sarajevo-specific [AQI scale](https://www.iqair.com/newsroom/what-is-aqi). Rather than serving as a decision-making system, the application functions as a proof of concept,
illustrating how model outputs can be explored, validated qualitatively, and communicated to users through intuitive visualizations and AQI-based interpretation.
### [Go to Air Quality Dashboard](https://airpollutionpredictionplatform-agbkast8wmll8pc6ntzhrk.streamlit.app/) 

## **Conclusion and future steps**
## **References**
Zolota, E., Hasić, V., Mević, A., Delić, A. & Krivić, S., 2024. Predictive Analysis of Sarajevo’s
AQI using Machine Learning Models for Varied Data Granularity and Prediction Windows. In:
2024 47th MIPRO ICT and Electronics Convention (MIPRO). IEEE, pp.1109–1114. [1](https://www.researchgate.net/publication/381818848_Predictive_Analysis_of_Sarajevo's_AQI_using_Machine_Learning_Models_for_Varied_Data_Granularity_and_Prediction_Windows)

Bekkar, A., Hssina, B., Douzi, S. & Douzi, K., 2021. Air-pollution prediction in smart city, deep
learning approach. Journal of Big Data, 8(1), p.161. [2](https://link.springer.com/article/10.1186/s40537-021-00548-1)

Thapa, I., Devkota, B., Lamichhane, B.R., Devkota, B.P., Dhakal, R. & Horanont, T., 2025.
Applying Deep Learning and Machine Learning Algorithms to Estimate PM2.5 Concentration Using
Satellite Data and Meteorological Data. IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing. [3](https://www.researchgate.net/publication/396549656_Applying_Deep_Learning_and_Machine_Learning_Algorithms_to_Estimate_PM_25_Concentration_Using_Satellite_Data_and_Meteorological_Data)

Chennareddy, S., Saha, S., Das, A. & Kayal, T., 2024. PM2.5 Concentration Forecasting in the
Kolkata Region With Spatiotemporal Sliding Window Approaches. IEEE Access, 12,
pp.82333–82353. [4](https://ieeexplore.ieee.org/abstract/document/10552267/)

Mazza, A., Guarino, G., Scarpa, G., Yuan, Q. & Vivone, G., 2025. PM2.5 retrieval with
Sentinel-5P data over Europe exploiting deep learning. IEEE Transactions on Geoscience and Remote
Sensing. [5](https://www.researchgate.net/publication/390287974_PM25_Retrieval_with_Sentinel-5P_Data_over_Europe_Exploiting_Deep_Learning)

Salcedo-Bosch, A., Zong, L., Yang, Y., Cohen, J.B. & Lolli, S., 2025. Forecasting particulate matter
concentration in Shanghai using a small-scale long-term dataset. Environmental Sciences Europe,
37(1), p.47. [6](https://link.springer.com/article/10.1186/s12302-025-01068-y)


## **License**
MIT License
