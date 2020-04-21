from evaluator.evaluator import Evaluator
from visual.plotter import plot_lines
from visual.reporter import get_report
import numpy as np
import pandas as pd
import json
import copy
import shutil
import time
import os


def evaluate(models,
             data_set_path,
             log_save_path,
             measurement,
             test_times=1):

    data_set_info_file = open(f"{data_set_path}\\data_set_info.json", 'r')
    data_set_info = json.load(data_set_info_file)

    if not os.path.exists(f"{log_save_path}\\{data_set_info['name']}"):
        os.mkdir(f"{log_save_path}\\{data_set_info['name']}")

    shutil.copy(f"{data_set_path}\\data_set_info.json", f"{log_save_path}\\{data_set_info['name']}")

    stream_list = [path for path in os.listdir(data_set_path) if os.path.isdir(f"{data_set_path}\\{path}")]
    stream_list.sort(key=lambda s: int(s.split('_')[1]))  # stream_number

    records_list = []
    for stream in stream_list:
        log_stream_path = f"{log_save_path}\\{data_set_info['name']}\\{stream}"
        if not os.path.exists(log_stream_path):
            os.mkdir(log_stream_path)
        if not os.path.exists(f"{log_stream_path}\\record"):
            os.mkdir(f"{log_stream_path}\\record")
        if not os.path.exists(f"{log_stream_path}\\figure"):
            os.mkdir(f"{log_stream_path}\\figure")
        data = pd.read_csv(f"{data_set_path}\\{stream}\\data.csv", header=None)
        with open(f"{data_set_path}\\{stream}\\data_info.json", 'r') as data_info_file:
            data_info = json.load(data_info_file)
        X = np.array(data.iloc[:, 1:])
        y = np.array(data.iloc[:, 0])

        perf_records = {}
        for model in models:
            for _ in range(test_times):
                test_model = copy.deepcopy(models[model])
                test_model.budget = data_info['budget']
                evaluator = Evaluator(measurement=measurement,
                                      pretrain_size=1,
                                      batch_size=1,
                                      budget=data_info['budget'])
                if model not in perf_records.keys():
                    perf_records[model] = evaluator.evaluate(X, y, model=test_model)
                else:
                    perf_records[model] += evaluator.evaluate(X, y, model=test_model)
            perf_records[model] = perf_records[model] / test_times
        perf_records = pd.DataFrame(perf_records)
        records_list.append(perf_records)
        perf_records.to_csv(f"{log_stream_path}\\record\\{measurement}.csv", index=None)
        plot_lines(perf_records, f"{log_stream_path}\\figure\\{measurement}", "pdf", 15, 'time', measurement)

    report_file = open(f"{log_save_path}\\report.md", 'a')
    report_file.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n")
    report_file.write(f"{data_set_info}\n\n")
    report_file.write(f"{measurement}\n\n")
    table = get_report(data_set_info, records_list, file_type="md")
    report_file.write(f"{table}\n")
    report_file.close()


