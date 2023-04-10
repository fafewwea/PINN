import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def data_preprocess(empirical, physics):

    data_total = pd.read_pickle('/home/yubinryu/ml/PINN_final/final_data/total.pkl')
    data_total = data_total.sample(empirical, random_state=42)

    data_in = pd.read_pickle('/home/yubinryu/ml/PINN_final/final_data/inner.pkl')
    data_in = data_in.sample(physics, random_state=42)

    data_out = pd.read_pickle('/home/yubinryu/ml/PINN_final/final_data/outer.pkl')
    data_out = data_out.sample(physics, random_state=42)

    data = pd.concat([data_total, data_in, data_out], axis=0, join='outer')
    data = data.drop_duplicates(subset=['x-velocity', 'y-velocity', 'z-velocity'], keep='first')

    del data_in
    del data_out
    del data_total

    train = data[['x-coordinate', 'y-coordinate', 'z-coordinate', 'input-concentration', 'input-temperature', 'domain']].to_numpy()
    target = data[['x-velocity', 'y-velocity', 'z-velocity', 'Pressure', 'Temperature', 'Initiator', 'Monomer']].to_numpy()

    interpolation_index = []

    for t in [360, 375, 390, 405, 420, 435]:
        for c in [0.00012, 0.00016, 0.00020, 0.00024, 0.00028, 0.00032]:
            index = np.reshape(np.round(train[:, 4:5], 5) == t, (train.shape[0],)) * np.reshape(
                np.round(train[:, 3:4], 5) == c, (train.shape[0],))
            idx_interpolation = np.array([i for i, x in enumerate(index) if x == True])
            interpolation_index += list(idx_interpolation)

    interpolation_train = train[interpolation_index, :]
    interpolation_target = target[interpolation_index, :]

    train = np.delete(train, interpolation_index, axis=0)
    target = np.delete(target, interpolation_index, axis=0)

    print(interpolation_train.shape)
    print(interpolation_target.shape)

    print(train.shape)
    print(target.shape)

    X_train, X_val, Y_train, Y_val = train_test_split(train, target, test_size=0.2, random_state=1)

    return X_train, X_val, Y_train, Y_val