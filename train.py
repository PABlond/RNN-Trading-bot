from core.model import Model
from core.data import DataModel
import json
import glob
from sklearn.model_selection import train_test_split
import pandas as pd

def check_dataset(dataset_path):
    if len(glob.glob(dataset_path)) < 1:
        print('{} is not available'.format(dataset_path))
        print('Please Download dataset from :')
        print('https://forextester.com/data/datasources')
        return False
    else:
        return True

def main():
    dataset_path = 'data/'
    # Requirement: Check the presence of the dataset
    if check_dataset(dataset_path):
        configs = json.load(open('config.json', 'r'))
        # 1) Build the model
        model = Model()
        model.build_model(configs)
        batch_size, epochs,  = 32, 3
        cols = configs['training']['cols']
        sequence_length = configs['data']['sequence_length']
        save_dir = "model"
        l = 0
        dataset_path = glob.glob("{}/*.txt".format(dataset_path))
        # 2 ) Loop over the files in the dataset folder 
        for filename in dataset_path:
            print("Training {}/{} - {}".format(l, len(dataset_path), filename))
            l += 1
            # 3) Divide the dataset in parts and loop over them
            chunksize = 10 ** 4
            for chunk in pd.read_csv(filename, chunksize=chunksize):
                # 4) Get and prepare data
                data = DataModel()
                x = data.get_train_data(
                    data=[x for x in chunk.get(cols).values.tolist()],
                    seq_len=sequence_length)
                X_train, X_test, y_train, y_test = train_test_split(data.dataX, data.dataY, test_size=0.33)
        
                # 5) Train the model       
                model.train(
                    X_train, X_test, y_train, y_test,
                    epochs = epochs,
                    batch_size = batch_size,
                    save_dir = save_dir)
        

if __name__ == '__main__':
    main()
