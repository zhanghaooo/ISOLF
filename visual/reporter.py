import pandas as pd


def nested_dict_get(d, key):
    value = d
    for key in key.split(':'):
        value = value[key]
    return value


def data_frame2md_table(df, value_columns=None, mark=None, header=True):
    table = ""
    if header:
        table += f"| {' | '.join(df.columns)} |\n"
        table += f"{'| --- ' * df.shape[1]}|\n"
    for i in range(df.shape[0]):
        mark_value = None
        if mark is not None:
            mark_value = eval(mark)(df.iloc[i].loc[value_columns])
        for j in range(df.shape[1]):
            if df.columns[j] in value_columns:
                if df.iloc[i, j] == mark_value:
                    table = table + f"| **{df.iloc[i, j]:.4f}** "
                else:
                    table = table + f"| {df.iloc[i, j]:.4f} "
            else:
                table = table + f"| {df.iloc[i, j]} "
        table = table + f"|\n"
    return table


def get_report(data_set_info, records_list, file_type="md"):
    test_params = pd.DataFrame(
        {data_set_info["test_param"]: nested_dict_get(data_set_info, data_set_info["test_param"])}
    )
    final_perf = pd.concat([records.iloc[-1:] for records in records_list],
                           ignore_index=True)
    instant_perf = pd.concat([pd.DataFrame(records.mean()).T for records in records_list],
                             ignore_index=True)

    if file_type == "md":
        table = data_frame2md_table(pd.concat([test_params, final_perf], axis=1),
                                    value_columns=records_list[0].columns,
                                    mark="max")
    return table

