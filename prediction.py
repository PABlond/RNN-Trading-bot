from core.model import Model
from core.data import DataModel
import json
from keras.models import load_model
import numpy as np
import fxcmpy
from keras import backend as K
from apscheduler.schedulers.blocking import BlockingScheduler

def request_and_predict(con, currency):
    configs = json.load(open("config.json", "r"))
    cols = configs['prediction']['cols']
    sequence_length = configs['data']['sequence_length']
    alphavantage_KEY = configs['data']['alphavantage_KEY']
    model = load_model('model/model.h5')
    diff = []
    for period in configs['prediction']['period']:
        prediction_len = period['prediction_len']
        # Get last 245 values
        data = DataModel()
        req = data.req_data(
            currency=currency, 
            cols=cols, 
            con=con,
            api_key=alphavantage_KEY,
            period=period['lapse'])
        if req:
            # Make a prediction of the next two values 
            x = data.get_predict_data(seq_len=sequence_length)
            # Calculate the profit from the prediction
            predictions = data.predict_sequences_multiple(model, x, sequence_length, prediction_len)
            next_pred = data.data_scaling_inv(predictions)[prediction_len-1][0]
            last_value = data.data_scaling_inv(x[0]).tolist()[248][0]
            diff.append(next_pred - last_value)
    return np.mean(diff)

def main():
    configs = json.load(open("config.json", "r"))
    fxcmpy_cfg = configs['data']['fxcmpy_cfg']
    con = fxcmpy.fxcmpy(config_file=fxcmpy_cfg)
    # Get currencies list
    currencies = con.get_instruments()[:20]
    # Loop over currencies
    for currency in currencies:
        # Close previous prediction for this currency
        con.close_all_for_symbol(currency)
        diff = request_and_predict(
            con=con,
            currency=currency)
        # Decide to buy or to sell from the difference
        print("Prediction of {} - {}".format(currency, diff))
        if diff > 0.2:
            print('BUYING: ', currency, int(diff*1000))
            con.create_market_buy_order(
                currency, 
                int(diff*1000))
        elif diff < -0.2:
            print('SELLING: ', currency, -1*int(diff*1000))
            con.create_market_sell_order(
                currency, 
                -1*int(diff*1000))  
    # Finally   
    con.close()
    K.clear_session()

if __name__ == '__main__':
    main()
    # Initialize a schedule
    scheduler = BlockingScheduler(timezone='MST')
    # Repeat main() every 2min
    scheduler.add_job(main, 'interval', id='my_job_id', seconds=540)
    scheduler.start()
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

    try:
        while True:
            time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()