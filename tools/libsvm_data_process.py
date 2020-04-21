import pandas as pd
import numpy as np
import shutil
import random
import json
import os


data_set_name = "mashroom"
data_path = "F://dataset//mashroom.txt"
stream_length = 100000
features = 123
X = np.zeros((stream_length, features))
y = np.zeros(stream_length)
data_file = open(data_path)
for i in range(stream_length):
    line = data_file.readline()
    if not line:
        break
    items = line.split(' ')
    y[i] = int(items[0])
    for j in range(1, len(items)):
        col_val = items[j].split(':')
        X[i, int(col_val[0]) - 1] = float(col_val[1])
y[y == y.min()] = 0
y[y == y.max()] = 1


def imbalance_sample(X, y, imbalance_rate):
    sampling_strategy = {key: value / max(imbalance_rate.values()) for key, value in imbalance_rate.items()}
    imbalance_X = []
    imbalance_y = []
    random.seed(a=666)
    for i in range(len(X)):
        if random.random() < sampling_strategy[y[i]]:
            imbalance_X.append(X[i, :])
            imbalance_y.append(y[i, :])
    return np.array(imbalance_X), np.array(imbalance_y)


# X, y = imbalance_sample(X, y, imbalance_rate={0: 40, 1: 1})
data = pd.concat([pd.DataFrame(y.reshape(-1, 1)).astype(int), pd.DataFrame(X)], axis=1)
imbalance_rate = data.iloc[:, 0].value_counts()

data_set_info = {
    "data_name": data_set_name,
    "stream_length": len(data),
    "n_features": features,
    "imbalance_rate": {0: imbalance_rate[0] / imbalance_rate[1], 1: 1},
    "test_params": ['imbalance_rate']
}
data_set_path = f"..//data//{data_set_name}"
if os.path.exists(data_set_path):
    shutil.rmtree(data_set_path)
os.mkdir(data_set_path)
data_set_info_file = open(f"{data_set_path}//data_set_info.json", 'w')
json.dump(data_set_info, data_set_info_file)
data_set_info_file.close()

data_info = {
    "data_name": data_set_name,
    "stream_length": len(data),
    "n_features": features,
    "imbalance_rate": {0: imbalance_rate[0] / imbalance_rate[1], 1: 1},
}
os.mkdir(f"{data_set_path}//stream_0")
data.to_csv(f"{data_set_path}//stream_0//data.csv", index=None, header=None)
data_info_file = open(f"{data_set_path}//stream_0//data_info.json", 'w')
json.dump(data_info, data_info_file)
data_info_file.close()



