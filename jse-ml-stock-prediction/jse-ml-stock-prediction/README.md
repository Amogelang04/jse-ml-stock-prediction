# JSE Stock Price Prediction using Machine Learning

This project builds a simple machine learning workflow to predict short-term stock price movements using historical data from the Johannesburg Stock Exchange (JSE).

I wanted this project to be practical and easy to explain, so I focused on a clean pipeline:
- download historical daily price data for a JSE-listed stock,
- create features from the price series,
- train a few baseline machine learning models,
- compare their performance,
- and visualise how the predictions behave on unseen data.

## Project idea

The main goal is to predict the **next day's closing price** using information from recent trading history.

The features include:
- daily returns,
- lagged returns,
- rolling moving averages,
- rolling volatility,
- momentum-style features,
- trading volume changes.

The models used are:
- Linear Regression
- Random Forest Regressor

This is not meant to be a trading system. It is a small data science and machine learning project that shows feature engineering, model training, evaluation, and visualisation in a finance context.

## Why this project makes sense

Stock prices are noisy, so it is unrealistic to claim perfect prediction. Because of that, I kept the project honest:
- the models are treated as baselines,
- the train/test split respects time order,
- performance is evaluated using regression metrics,
- and a simple directional accuracy measure is also reported.

## Folder structure

```text
jse-ml-stock-prediction/
│
├── data/                   # downloaded CSV data saved here
├── notebooks/              # optional notebook work
├── results/                # plots and metrics written here
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── plotting.py
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md
```

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/Amogelang04/jse-ml-stock-prediction.git
cd jse-ml-stock-prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Mac/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the training script

```bash
python src/train.py
```

## Default setup

The script currently uses:
- ticker: `SBK.JO` (Standard Bank Group)
- training window: historical daily data from 2018 onward
- target: next-day closing price

You can change the ticker inside `src/train.py` if you want to test another JSE stock such as:
- `NPN.JO`
- `AGL.JO`
- `SOL.JO`
- `FSR.JO`

## Outputs

After running the project, the script saves:
- cleaned historical data in `data/`
- model comparison metrics in `results/model_metrics.csv`
- prediction plots in `results/`

## Example evaluation metrics

The project reports:
- MAE
- RMSE
- R²
- Directional Accuracy

## Possible next steps

A few realistic improvements for future work:
- try classification for up/down movement,
- use walk-forward validation,
- add technical indicators such as RSI or MACD,
- compare more models such as XGBoost,
- include macroeconomic or market index features.

## Disclaimer

This project is for learning and portfolio purposes only. It is not financial advice and it should not be used on its own for investment decisions.
