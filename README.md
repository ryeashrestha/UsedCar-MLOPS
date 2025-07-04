# Used Car Price Prediction System

**Used Car Price Prediction System** — a fully automated machine learning project developed to estimate used car prices using real-world data and an end-to-end **MLOps pipeline**.

This project was built as part of the **CMP5366 - Data Management and Machine Learning Operations** module at **Birmingham City University**.

---

## Project Overview

Predicting used car prices is a real-world challenge affected by multiple variables like brand, model year, mileage, and fuel type. This system uses data scraped from cars.com to build a regression model that delivers accurate price predictions. The entire process is built and deployed using MLOps practices — covering ingestion, validation, transformation, modeling, deployment, and monitoring.


---

## Tech Stack & Tools Used

- **Languages & Libraries**: Python, Pandas, Scikit-learn, XGBoost, SQLAlchemy
- **Machine Learning**: XGBoost with Optuna hyperparameter tuning
- **Data Validation**: Great Expectations
- **Orchestration**: Apache Airflow
- **Model Serving**: FastAPI
- **Monitoring**: MLflow
- **Data Storage**: MariaDB (inside Docker), Redis
- **Environment**: Docker, Anaconda

---

## Dataset

- Source: Kaggle  
  [Used Car Price Prediction Dataset](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset)
- Size: 4,009 car listings
- Features: Brand, Model, Model Year, Mileage, Fuel Type, Engine, Transmission, Color, Condition, Price (Target)

---

---

## Project Features

- Built star schema for structured storage and efficient access
- Automated ETL pipeline to handle inconsistent, real-world data
- Data validation before and after processing using Great Expectations
- Feature engineering, encoding, standardization, and drift handling
- ML model trained using XGBoost and tuned with Optuna
- Model deployed via FastAPI and monitored with MLflow
- Plans for automated retraining, drift detection, and cloud deployment

---


