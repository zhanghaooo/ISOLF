from generator.hyperplane_stream_generator import HyperplaneStreamGenerator
import pandas as pd
import copy
import json
import shutil
import os


def listdict2dictlist(listdict):
    dictlist = [{}]
    for key in listdict.keys():
        dictlist_ = []
        for d in dictlist:
            if isinstance(listdict[key], dict):
                sub_dictlist = listdict2dictlist(listdict[key])
                for value in sub_dictlist:
                    d_ = copy.deepcopy(d)
                    d_[key] = value
                    dictlist_.append(d_)
            elif isinstance(listdict[key], list):
                for value in listdict[key]:
                    d_ = copy.deepcopy(d)
                    d_[key] = value
                    dictlist_.append(d_)
            else:
                d_ = copy.deepcopy(d)
                d_[key] = listdict[key]
                dictlist_.append(d_)
        dictlist = copy.deepcopy(dictlist_)
    return dictlist


def data_set_generate(data_set_config, save_path="..\\data"):
    data_set_path = f"{save_path}\\{data_set_config['name']}"
    if os.path.exists(data_set_path):
        shutil.rmtree(data_set_path)
    os.mkdir(data_set_path)
    stream_configs = listdict2dictlist(data_set_config)

    for stream_config in stream_configs:
        stream_generator = HyperplaneStreamGenerator(**stream_config["generate_config"])
        X, y = stream_generator.get_samples(stream_config["stream_length"])
        data = pd.concat([pd.DataFrame(y), pd.DataFrame(X)], axis=1)
        stream_config["stream_length"] = len(data)
        new_stream_path = f"{data_set_path}\\stream_{len(os.listdir(data_set_path))}"
        os.mkdir(new_stream_path)
        data.to_csv(f"{new_stream_path}\\data.csv", index=False, header=False)
        data_info_file = open(f"{new_stream_path}\\data_info.json", 'w')
        json.dump(stream_config, data_info_file)
        data_info_file.close()

    data_set_info_file = open(f"{data_set_path}\\data_set_info.json", 'w')
    json.dump(data_set_config, data_set_info_file)
    data_set_info_file.close()
