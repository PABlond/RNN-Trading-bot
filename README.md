WIP
# Information
Stock Price Prediction with Recurrent Neural Networks (RNN) mostly built with Keras.

Input data is stored in the data/ directory. You'll notice that there is an example dataset included in the repo which consists of a subset EUR/USD exchange rate. Full versions of this dataset can be find here : https://forextester.com/data/datasources


# Installation Guide

## Step 1 : clone this repository

    git clone https://github.com/PABlond/RNN_Forex-trading-bot.git
    
## Step 2 : install dependencies

    cd RNN_Forex-trading-bot/    
    pip install sklearn pandas keras numpy fxcmpy apscheduler
    
## Step 3 : Get an API token from FXCM 

To connect to the API, you need an API token that you can create or revoke from within your (demo) account in the Trading Station https://tradingstation.fxcm.com/.
Then, you will need to paste your token in the fxcm config file (core/cfg/fxcm.cfg)

    [FXCM]
    log_level = error
    log_file = fxcm.log
    access_token = <YOUR_FXCM_TOKEN>

## Step 4 : train the model

    python train.py

## Step 5 : launch the bot

    python prediction.py
    
# Contributor:
@PABlond
