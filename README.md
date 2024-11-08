# Intrade: Predicting Stock Price using WGAN-GP
## NOTE
_this repository is maintained under https://gitlab.com/majimeniukiyo/intrade_predict_gan_
_for any inquiry, please submit an issue on the provided [link](https://gitlab.com/majimeniukiyo/intrade_predict_gan.git)_

# Introduction

This repository implements the prediction of the stock price using WGAN-GP, based on the paper *[“Improved Training of Wasserstein GANs”](https://arxiv.org/abs/1704.00028)* by the authors ***********************Gulrajani et al., 2017.***********************

This repository is inspired by https://github.com/borisbanushev/stockpredictionai and [https://keras.io/examples/generative/wgan_gp](https://keras.io/examples/generative/wgan_gp/)

# Requirements

- python 3.8-3.10
- Linux
- Conda

## Conda Environment

Run the following command in the terminal to install and activate the conda environment

```python
conda create environment.yaml
```

## Initializing Tensorflow GPU

Run the following command in the terminal to initialize Tensorflow GPU

```python
sudo source path/to/tensorflow_gpu_init.sh
```

# Preparing Data

## Directories

- root
    - data
        - aapl
            - aapl_dataset_arima.csv
            - aapl_dataset_fourier.csv
            - aapl_dataset_important.csv
            - aapl_dataset_indices.csv
            - aapl_dataset_sym.csv
            - aapl_dataset_technical.csv
            - aapl_dataset.csv
    - predictions
        - stock_price
            - aapl_predictions.csv

# Preparing Assets

## Directories

- root
    - assets
        - models.yaml
        - hyperparameters.json

## Models.yml

```yaml
models:
  aapl:
    RMSE: 0.0
    features:
    - close-^BSESN
    - close-^IXIC
    - close-^N225
    - close-^RUT
    - absolute of 100 comp
    - arima
    - SMA7
    - EMA7
    - EMA21
    - lower_band
    model_path: ''
    no_features: 10
    predicted_days: 7
    sym: aapl
    timesteps: 50
    training_data_first: '2010-01-04'
    training_data_last: '2023-07-03'
```

## Hyperparameters.json

```json
{
    "opt" : [
        {
            "normalization_technique":"keras_norm",
            "train_split" : "percentage",
            "train_size_percentage" : 0.75,
            "val_size_percentage" : 0.15,
            "epoch": 1500, 
            "bs": 64, 
            "discriminator_steps": 5, 
            "generator_steps" : 1,
            "run_type" : "test_market_",
            "model_name" : "intrade_gan_model_"
        }
    ],

    "fixed_params" : [
        {
            "gen_init_trucn_mean" : 0.0,
            "g_learning_rate" : 0.00005, 
            "g_max_learning_rate" : 0.001, 
            "g_learning_rate_update" : true, 
            "g_learning_rate_scheduler" : "cycle"
        }
    ],

    "hyper_params" : [
        {
            "gen_init_trunc_stdev" : 0.01,
            "d_learning_rate" : 0.001, 
            "gp_weight" : 30, 
            "recurrent_dropout" : 0.02, 
            "gen_l1_regulizer" : 1e-3, 
            "bn_momentum" : 0.99, 
            "conv_dropout" : 0.2, 
            "conv_lrelu_activation" : 0.01, 
            "conv_l2_regulizer" : 1e-3, 
            "dense_l2_regulizer" : 1e-3, 
            "dense_dropout" : 0.5, 
            "dense_bias_init" : 0.1, 
            "dense_lrelu_activation" : 0.01
        }
    ],

    "hyper_params_pbounds" : [
        {
            "gen_init_trunc_stdev" : [0.01 , 0.5],
            "d_learning_rate" : [0.00005, 0.001], 
            "gp_weight" : [5 , 30],
            "recurrent_dropout" : [0.01 , 0.02], 
            "gen_l1_regulizer" : [1e-3, 0.01], 
            "bn_momentum" : [0.1, 0.99], 
            "conv_dropout" : [0.01 , 0.5], 
            "conv_lrelu_activation" : [0.0001, 0.1], 
            "conv_l2_regulizer" : [1e-3, 0.1], 
            "dense_l2_regulizer" : [1e-3, 0.1], 
            "dense_dropout" : [0.01 , 0.8], 
            "dense_bias_init" : [0.1 , 0.9], 
            "dense_lrelu_activation" : [0.0001, 0.1]}
    ]
}
```

# Using WGAN-GP to Predict Prices

## Running the Prediction

```python
python run.py
```

## Command Line Arguments

```python
--sym: Ticker symbol to run prediction on, default='aapl'
--data_dir: Path to the stock data, default='intrade/ticker/data/'
--predictions_dir: Path to the stock output predictions default='intrade/ticker/predictions/stock_price/'
--model_dir: Path to the generator model, default='intrade/predict/intrade_predict_gan/output/models/'
--hyp_dir: Path to the hyperparameters file, default='intrade/predict/intrade_predict_gan/assets/hyperparameters.json'
--symbol_models_dir: Path to the symbols models configuration file, default='intrade/predict/intrade_predict_gan/assets/models.yml'
--load_training_data: Import training data from a directory, default=False
--load_denoised_data: Load ARIMA and FFT data ONLY if they are available and up to date. If load_training_data is true then this parameter is ignored, default=False
--reload_features: If reload_features is True, the method will recompute the feature importance., default=False
--timesteps: Number of timesteps to be used for training. Predict the stock price based on the last 7 days, default=7
--predicted_days: Number of days to be predicted. Predict the stock price for the next day/s, default=1
```
