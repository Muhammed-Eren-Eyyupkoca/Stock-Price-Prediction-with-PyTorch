# Stock Price Prediction with PyTorch (LSTM & GRU)

## Proje Amacı

Bu proje, PyTorch kullanarak geçmiş hisse senedi verilerinden gelecekteki fiyat hareketlerini tahmin etmeyi amaçlayan bir zaman serisi tahmin uygulamasıdır. LSTM ve GRU mimarilerinin performansını karşılaştırarak hisse senedi fiyat tahmininde hangi modelin daha başarılı olduğunu ortaya koymayı hedefler.

## Description

A time-series stock price prediction pipeline built with PyTorch. This project fetches historical stock data, performs exploratory data analysis, and trains deep learning sequence models to forecast future prices. The main goal is a **comparative study of LSTM vs. GRU architectures**, evaluating which recurrent model performs better on stock price forecasting.

## 🚀 Project Roadmap (4 Weeks)

### 📊 Week 1: Data Acquisition & Exploratory Data Analysis (EDA)
- [x] Set up the GitHub repository structure and virtual environment.
- [x] Fetch historical stock data using Yahoo Finance (yfinance) or pandas-datareader.
- [x] Perform Exploratory Data Analysis (EDA) using Pandas and Matplotlib/Seaborn.
- [x] Deliverable: Jupyter Notebook with data visualizations and clean, saved CSV datasets.

### ⚙️ Week 2: Data Preprocessing & Baseline LSTM Model
- [x] Prepare the dataset for time-series (scaling, sequence creation, train-test split).
- [x] Create PyTorch Dataset and DataLoader classes.
- [x] Build and train a baseline LSTM model in PyTorch.
- [x] Deliverable: Working PyTorch training pipeline with training loss visualization.

### 🧠 Week 3: GRU Model Implementation & Hyperparameter Tuning
- [x] Implement the GRU model architecture in PyTorch.
- [ ] Train the GRU model on the same prepared dataset.
- [ ] Experiment with hyperparameters (learning rate, hidden dimensions, epochs).
- [ ] Deliverable: Trained LSTM and GRU model weights saved locally.

### 📈 Week 4: Model Evaluation, Comparison & Final Polish
- [ ] Evaluate both models on test data using metrics like RMSE, MSE, and MAE.
- [ ] Plot predictions vs. actual stock prices for both models.
- [ ] Document final results, limitations, and key learnings in the README.md.
- [ ] Deliverable: Fully completed GitHub repository, clean notebook, and a comparison table.

## Project Structure

```
├── data/                       # Raw and processed datasets (ignored by git)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── dataset.py              # Data loading & preprocessing
│   ├── models.py                # LSTM & GRU model definitions
│   └── train.py                 # Training loop
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```
