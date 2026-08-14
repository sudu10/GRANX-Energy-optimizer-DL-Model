# GRANX-Energy-optimizer-DL-Model

## GRAN-X: Unified Deep Learning Model for Energy Forecasting and Optimization

GRAN-X (Gradient-Regulated Attention Network) is a hybrid deep learning framework designed for **Short-Term Load Forecasting (STLF)** and intelligent household energy optimization.

The system combines **multi-scale CNN feature extraction, dual GRU/LSTM temporal modeling, multi-head attention, gradient-based residual correction, and Particle Swarm Optimization (PSO)** to forecast energy consumption and generate cost-aware appliance schedules.

## Overview

Traditional forecasting models often struggle with sudden appliance load spikes, long-term temporal dependencies, changing tariffs, and cumulative prediction errors. GRAN-X addresses these limitations by combining complementary neural architectures into a unified forecasting pipeline.

The complete workflow is:

```text
Smart Meter + Appliance + Context Data
                │
                ▼
       Data Preprocessing
                │
                ▼
      Feature Engineering
                │
                ▼
       Conv1D Feature Extraction
                │
                ▼
        Dual GRU + LSTM
                │
                ▼
       Multi-Head Attention
                │
                ▼
      Gradient Correction
                │
                ▼
       Energy Forecast
                │
                ▼
      PSO Optimization
                │
                ▼
     Optimized Appliance
          Scheduling
```

The architecture is intended to support household energy management, demand response, peak-load management, dynamic pricing, and cost-efficient scheduling.

## Key Features

* Short-Term Load Forecasting
* Multivariate time-series processing
* Multi-scale temporal feature extraction
* Conv1D-based local pattern detection
* Dual GRU/LSTM temporal modeling
* Multi-Head Self-Attention
* Gradient-based residual correction
* Particle Swarm Optimization for scheduling
* Appliance-level energy analysis
* Tariff-aware optimization
* Attention-based interpretability
* MAE, MSE, and RMSE evaluation
* Time-series chronological train/test splitting

## Model Architecture

### 1. Data Conditioning

GRAN-X accepts multivariate energy data containing variables such as:

* Timestamp
* Total household power
* Appliance-level consumption
* Occupancy
* Weather/season
* Time-of-day slot
* Tariff
* Energy cost

The model applies:

* `StandardScaler` to input features
* `MinMaxScaler` to the prediction target
* Sliding-window sequence generation
* Lag features
* Rolling statistics
* Cyclic temporal features

The documented feature engineering includes lag values such as `t-1`, `t-24`, and `t-168`, together with 4-hour and 24-hour rolling statistics and temporal encodings.

### 2. Conv1D Feature Extraction

The CNN stage extracts short-term and localized temporal patterns such as:

* Appliance activation spikes
* Daily consumption patterns
* Short-duration fluctuations
* Local appliance interactions

The documented architecture uses sequential Conv1D layers with ReLU activation, Batch Normalization, and Dropout. The model configuration specifies 32 and 64 filters with a kernel size of 3 and dropout of 0.2.

### 3. Dual GRU + LSTM Temporal Modeling

The convolutional feature representation is passed into parallel recurrent components:

* **GRU** - captures shorter and medium-term temporal dependencies
* **LSTM** - captures longer-term dependencies

Their outputs are concatenated to create a richer temporal representation of household energy behavior.

### 4. Multi-Head Attention

The attention mechanism dynamically assigns importance to different historical time steps.

The documented configuration uses **8 attention heads** and computes attention from Query, Key, and Value representations. This allows GRAN-X to focus on periods that have greater influence on future consumption.

For example, the study reports higher attention relevance during **18:00–22:00**, corresponding to increased occupancy and appliance activity.

### 5. Gradient Correction Layer

The Gradient Correction Layer is the core innovation of GRAN-X.

Instead of relying only on conventional backpropagation, the architecture introduces a gradient-based correction mechanism in the prediction pipeline to reduce residual prediction error.

The documented correction rate is `0.01`.

Conceptually:

```text
Attention Context
       │
       ▼
 Initial Prediction
       │
       ▼
 Estimate Loss Gradient
       │
       ▼
 Gradient-Based Correction
       │
       ▼
 Refined Prediction
```

### 6. Particle Swarm Optimization

PSO extends the forecasting system from prediction to actionable energy optimization.

Each particle represents a possible appliance scheduling configuration. The optimizer searches for schedules that minimize energy cost while respecting appliance operating constraints and the predicted load profile.

The optimization can shift high-power appliance operation toward lower-cost tariff periods.

## Dataset

The model works with high-resolution multivariate household energy data.

The documented dataset contains:

| Feature       | Type        | Description                        |
| ------------- | ----------- | ---------------------------------- |
| `timestamp`   | Temporal    | Measurement timestamp              |
| `total_power` | Continuous  | Total household energy consumption |
| `appliances`  | Continuous  | Appliance-level consumption        |
| `occupancy`   | Categorical | Household presence indicator       |
| `weather`     | Categorical | Seasonal/climate condition         |
| `tod_slot`    | Categorical | Peak/off-peak time slot            |
| `tariff`      | Categorical | Energy pricing period              |
| `cost`        | Continuous  | Computed energy cost               |

The analysis also identified strong relationships between several appliance loads and total consumption. Washing machine, dryer, and kettle showed correlations of approximately `0.72–0.85` with total power, while occupancy showed a correlation of `0.58`.

## Forecasting Objective

The primary forecasting objective is to predict **future household energy consumption**, particularly the next time interval, from historical consumption and contextual information.

The model therefore learns:

```text
Historical Energy Usage
        +
Appliance Usage
        +
Occupancy
        +
Weather
        +
Time Information
        +
Tariff Information
        │
        ▼
      GRAN-X
        │
        ▼
Future Energy Consumption
```

## Optimization Pipeline

After forecasting, GRAN-X feeds predicted appliance demand into the PSO optimization stage.

```text
GRAN-X Forecast
      │
      ▼
Predicted Appliance Loads
      │
      ▼
PSO Scheduling
      │
      ├── Appliance constraints
      ├── Operating duration
      ├── Scheduling windows
      ├── Tariff periods
      └── Load balance
      │
      ▼
Optimized Appliance Schedule
```

The documented PSO scheduler converged within **20 iterations** in the reported experiment.

## Model Evaluation

GRAN-X was compared against:

* RNN
* GRU
* CNN
* XGBoost
* CNN-GRU

The reported percentage reductions are:

| Model      | RMSE Reduction | MSE Reduction | MAE Reduction |
| ---------- | -------------: | ------------: | ------------: |
| RNN        |          10.1% |          7.9% |          8.5% |
| GRU        |          12.6% |          9.8% |         11.4% |
| CNN        |           8.7% |          6.5% |          7.2% |
| XGBoost    |          17.4% |         13.9% |         15.6% |
| CNN-GRU    |          21.7% |         18.4% |         19.3% |
| **GRAN-X** |      **32.4%** |     **28.6%** |     **26.8%** |

These values are reported relative to the RNN baseline.

The reported evaluation also found that removing the attention module increased RMSE by 9.2%, removing the GRU block increased prediction variance by 11.7%, and removing the XGBoost refinement increased MSE by 15.5%.

## Interpretability

GRAN-X provides two complementary sources of interpretability:

### Attention Analysis

Attention weights indicate which historical time periods had the greatest influence on the forecast.

The reported results show stronger relevance during evening periods between **18:00 and 22:00**.

### Feature Importance

The reported feature analysis identified:

* Weather
* Tariff
* Occupancy

as important contextual drivers of household energy consumption.

## Expected Project Structure

A recommended repository structure is:

```text
GRANX-Energy-optimizer-DL-Model/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── granx_model/
│   ├── checkpoints/
│   └── scalers/
│
├── preprocessing/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   └── sequence_generator.py
│
├── forecasting/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── optimization/
│   ├── pso_optimizer.py
│   └── scheduler.py
│
├── visualization/
│   ├── attention_plots.py
│   ├── forecasting_plots.py
│   └── optimization_plots.py
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── LICENSE
```

> **Note:** The structure above is a recommended repository organization and is not a file structure specified in the research document.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd GRANX-Energy-optimizer-DL-Model
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train the Model

```bash
python forecasting/train.py
```

### Generate Forecasts

```bash
python forecasting/predict.py
```

### Evaluate the Model

```bash
python forecasting/evaluate.py
```

### Run Energy Optimization

```bash
python optimization/pso_optimizer.py
```

> The commands above assume the recommended project structure and should be adjusted to match the actual implementation.

## Key Technologies

```text
Python
Deep Learning
Conv1D
GRU
LSTM
Multi-Head Attention
Gradient-Based Error Correction
Particle Swarm Optimization
Time-Series Forecasting
Energy Analytics
Smart Grid Optimization
```

## Applications

GRAN-X can be applied to:

* Smart Home Energy Management Systems (HEMS)
* Short-Term Load Forecasting
* Demand Response
* Peak Load Management
* Dynamic Tariff Optimization
* Appliance Scheduling
* Energy Cost Reduction
* Smart Grid Analytics
* Edge/Cloud Energy Management
* Renewable Energy Integration

The research describes the architecture as modular and potentially scalable from individual household forecasting to cloud-integrated grid-level systems.

## Research Contributions

The main contributions of GRAN-X are:

1. **Multi-scale feature extraction** using convolutional layers.
2. **Dual temporal modeling** through GRU and LSTM networks.
3. **Dynamic temporal prioritization** using multi-head attention.
4. **Gradient-based residual correction** for prediction refinement.
5. **PSO-based optimization** for appliance scheduling.
6. **Tariff-aware energy optimization** connecting forecasting with cost reduction.
7. **Interpretable forecasting** through attention weights and feature importance.
8. **Modular architecture** designed for scalable smart energy applications.

The overall design was developed to bridge the gap between energy forecasting and actionable energy optimization.

## Results Summary

According to the reported experiments, GRAN-X achieved:

* **32.4% RMSE reduction**
* **28.6% MSE reduction**
* **26.8% MAE reduction**
* Lower prediction error than the evaluated RNN, GRU, CNN, XGBoost, and CNN-GRU baselines
* Stable performance across evaluation intervals
* PSO scheduling convergence within 20 iterations

## Future Work

Potential future extensions identified in the research include:

* Probabilistic energy forecasting
* Uncertainty quantification
* Real-time energy data streams
* Adaptive online learning
* Larger grid-level datasets
* Edge-based deployment
* Integration with renewable energy systems
* More advanced demand-response optimization

## Citation

If you use GRAN-X in academic work, cite the associated research document:

```bibtex
@article{granx_energy_forecasting,
  title   = {GRANX: A Unified Deep Learning Model for Energy Forecasting and Optimization},
  author  = {D Sudarsanan},
  year    = {2026},
  note    = {Gradient-Regulated Attention Network for Short-Term Load Forecasting and Energy Optimization}
}
```

## License

Add the project's chosen license here, for example:

```text
MIT License
```

If no license has been selected yet, do not assume that the repository is open-source licensed.
