def dict2md_list(d):
    s = ""
    for key in d.keys():
        s = s + f"+ {key}: "
        if isinstance(d[key], dict):
            sub_items = dict2md_list(d[key]).splitlines()
            sub_items = [f"  {item}\n" for item in sub_items]
            sub_s = ""
            for item in sub_items:
                sub_s = sub_s + item
            s = s + f"\n{sub_s}"
        else:
           s = s + str(d[key]) + "\n"
    return s


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
