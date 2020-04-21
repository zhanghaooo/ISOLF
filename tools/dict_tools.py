import copy


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


def nested_dict_get(dict, key):
    value = dict
    for key in key.split(':'):
        value = value[key]
    return value


def list_items_dectector(d):
    list_params = []
    for key in d:
        if isinstance(d[key], list) and len(d[key]) > 1:
            list_params.append(key)
        if isinstance(d[key], dict):
            sub_keys = list_items_dectector(d[key])
            for sub_key in sub_keys:
                list_params.append(f"{key}:{sub_key}")
    return list_params
