# 🚌 Bus Seat Demand Forecasting using LSTM

## 📌 Overview

This project implements a **time-series forecasting model** to predict **bus seat demand** based on historical booking and search data. The model leverages an **LSTM (Long Short-Term Memory)** network to capture temporal dependencies in past transactions and forecast final seat counts.

The dataset comes from **RedBus / Analytics Vidhya challenge**, containing millions of transactions with features such as:

* Source & destination IDs
* Date of journey (DOJ)
* Cumulative seat bookings and search counts for the last 15 days (`dbd` = days before departure)

---

## ⚙️ Project Workflow

1. **Data Loading & Preprocessing**

   * Read training, test, and transaction datasets.
   * Construct pivot tables for past 15 days (`cumsum_seatcount`, `cumsum_searchcount`).
   * Merge with training/test data and reshape into **3D arrays**: `(samples, timesteps=15, features=2)`.
   * Features scaled using `StandardScaler`.

2. **Model Architecture**

   * Framework: TensorFlow / Keras
   * Layers:

     * `LSTM(50, activation='relu')`
     * `Dropout(0.2)`
     * `Dense(1)` for regression output
   * Optimizer: Adam (lr=0.001)
   * Loss: Mean Squared Error (MSE)

3. **Training & Validation**

   * Train/validation split: 80/20 (no shuffle to preserve temporal order).
   * Model trained for 30 epochs with `ModelCheckpoint` saving best weights.
   * Training & validation loss curves plotted.

4. **Evaluation Metrics**

   * Root Mean Squared Error (RMSE)
   * Relative / Normalized RMSE (`RMSE / mean(y)`) for better interpretability.
   * Achieved **Relative RMSE ~0.17** on test data.

---

## 📊 Results

* **Relative RMSE (Train)**: **0.19**
* **Relative RMSE (Test)**: **0.17**

The LSTM successfully models sequential booking/search trends and provides robust demand forecasts.
