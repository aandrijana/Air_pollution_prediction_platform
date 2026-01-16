# Air pollution prediction platform (Sarajevo)

Collaborators: Andrijana Kešelj, Anđela Maksimović and Šehzada Sijarić

This project is done by the  students of DSAI at ETF Sarajevo.

## Table of Contents
- [📖 Overview](#overview)
- [🌫️ Problem: Air Pollution in Sarajevo](#problem-air-pollution-in-sarajevo)
- [🎯 Our Objective](#our-objective)
- [🧩 Commonly Used Terms](#commonly-used-terms)
- [📊 Dataset Choice](#dataset-choice)
- [🧹 Data Preparation & EDA](#data-preparation--eda)
- [🧠 Our Initial Approach](#our-initial-approach)
- [📈 Initial Results and Evaluation](#initial-results-and-evaluation)
- [⚙️ Project Challenges](#project-challenges)
- [🔧 Improvements](#improvements)
- [🏁 Final Results](#final-results)
- [💻 Platform Development](#platform-development)
- [🚀 Our Final Product!](#our-final-product)
- [🔮 Conclusion and Future Steps](#conclusion-and-future-steps)
- [📚 References](#references)
- [📄 License](#license)

## 📖 **Overview**
This project focuses on predicting air pollution levels in Sarajevo at the neighborhood level using machine learning models. To capture different characteristics of the data, we applied a GRU-based time-series model for PM10 prediction and an XGBoost model for PM2.5 prediction. The overall goal is to provide reliable short-term predictions (7-days) that can support air quality monitoring, and are able to inform citizens about AQI and its harmful health-related consequences.
## 🌫️ **Problem: Air pollution in Sarajevo**
Air pollution has been a long-standing issue in Sarajevo, with serious consequences for public health and the environment. The problem is driven by factors such as industrial emissions, heavy traffic, and the city’s geographical location. During winter months, temperature inversions trap pollutants near the ground, often turning Sarajevo into a smog-filled city and significantly worsening air quality.

As a result, Sarajevo is frequently ranked among the most polluted cities in the world, with other cities across Bosnia and Herzegovina facing similar challenges. Prolonged exposure to polluted air poses major health risks, including respiratory and cardiovascular diseases, particularly for vulnerable groups such as children, the elderly, and individuals with pre-existing health conditions.
## 🎯 **Our objective**
The goal of this project is to develop an ML-powered air quality prediction platform capable of forecasting PM2.5 and PM10 levels at the neighborhood level in Sarajevo, with potential future expansion to other cities.

The platform aims to provide personal exposure alerts and health-focused recommendations, helping residents take precautionary measures and supporting actions to improve air quality.

## 🧩 **Commonly used terms**
## 📊 **Dataset choice**
We used publicly available air quality data from reference monitoring stations located across multiple neighborhoods in Sarajevo (e.g., Otoka, Bjelave, U.S. Embassy, Ilijaš, etc..)

The dataset can be found on this [link](https://aqicn.org/map/sarajevo/)
We concatenated the datasets of all stations into one large dataset.
Different station datasets start at different times, but all data end at a common cutoff date of November 19, 2025.
## 🧹 **Data preparation & EDA**
## 🧠 **Our initial approach**
## 📈 **Initial results and evaluation**
## ⚙️ **Project challenges**
## 🔧 **Improvements**
## 🏁 **Final results**
## 💻 **Platform development**
## 🚀 **Our final product!**
## 🔮 **Conclusion and future steps**
## 📚 **References**
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


## 📄 **License**
MIT License
