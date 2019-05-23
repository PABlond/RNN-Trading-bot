from core.model import Model
from core.data import DataModel
import json
from keras.models import load_model
import numpy as np
import fxcmpy
from keras import backend as K
from apscheduler.schedulers.blocking import BlockingScheduler

def main():
    configs = json.load(open("config.json", "r"))
    cols = configs['prediction']['cols']
    sequence_length = configs['data']['sequence_length']
    alphavantage_KEY = configs['data']['alphavantage_KEY']
    fxcmpy_cfg = configs['data']['fxcmpy_cfg']
    con = fxcmpy.fxcmpy(config_file=fxcmpy_cfg)
    # Get currencies list
    currencies = con.get_instruments()[:5] 
    model = load_model('model/model.h5')
    # Loop over currencies
    for currency in currencies:
        # Get last 245 values
        data = DataModel()
        # Close previous prediction for this currency
        con.close_all_for_symbol(currency)
        req = data.req_data(
            currency=currency, 
            cols=cols, 
            con=con,
            api_key=alphavantage_KEY)
        if req:
            # Make a prediction of the next two values 
            print("Prediction of {}".format(currency))
            x = data.get_predict_data(seq_len=sequence_length)
            # Calculate the profit from the prediction
            predictions = data.predict_sequences_multiple(model, x, sequence_length, 2)
            next_pred = data.data_scaling_inv(predictions)[1][0]
            last_value = data.data_scaling_inv(x[0]).tolist()[248][0]
            diff = next_pred - last_value
            # Decide to buy or to sell from the difference
            if diff > 0.02:
                print('BUYING: ', currency, int(diff*100))
                con.create_market_buy_order(
                    currency, 
                    int(diff*100)
                    )
            elif diff < -0.02:
                print('SELLING: ', currency, -1*int(diff*100))
                con.create_market_sell_order(
                    currency, 
                    -1*int(diff*100)
                    )  
    # Finally   
    con.close()
    K.clear_session()

if __name__ == '__main__':
    main()
    # Initialize a schedule
    scheduler = BlockingScheduler(timezone='MST')
    # Repeat main() every 2min
    scheduler.add_job(main, 'interval', id='my_job_id', seconds=120)
    scheduler.start()
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

    try:
        while True:
            time.sleep(5)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()