# ISOLF

# 项目结构
+ data 测试数据集存放路径 （需要手动创建该路径）
+ models 存放模型的路径
+ tools 工具文件夹
    + data_generator.py：自动生成测试数据集
    + DATA_SETS_CONFIG：存放 data_generator.py 所需的数据集配置
    + data_show.py：二维流数据的动态显示工具
    + dict_tools.py：对python dict 的一些拓展功能
    + python2markdown.py：将 python 里的数据结构按照 markdown 的语法打印，用于实验报告的自动化生成
    + record_analyzer.py：将 scikit-multiflow 的 evaluator 输出解析成 python 数据结构
+ test 实验脚本
    + test.py 自动实验脚本
    + log.md 实验得到的测试结果会自动写入该日志文件。


# data_generator.py 使用说明

data_generator 一次可生成一系列相关数据集，用于对具有不同特性值的相关数据集进行对比实验。

Example：如果想测试一下各个模型在不同的正负样本比例及噪音情况下的表现情况如何。则需要生成一系列数据集来做对照试验，它们拥有不同的正负样本比例及噪音比例，但其他属性均相同。

```python
DataSets = {
    # 想要生成的相关数据集集合的名字（需要保持唯一）
    'name': 'noise-imbalance',
    
    # 使用 scikit-multiflow 中的 HyperplaneGenerator 生成此类数据集，必须是 scikit-multiflow 中的某种 stream generator 
    'data_type': 'HyperplaneGenerator', 

    # 流配置，HyperplaneGenerator 的参数，与 scikit-multiflow 中定义的该 stream generator 的参数保持一致
    'stream_configs': {
        'n_features': [2],
        'n_drift_features': [0],
        'mag_change': [0],
        'noise_percentage': [0, 0.01, 0.02, 0.05],
        'sigma_percentage': [0]
    },
    
    # 生成的每个数据集中包含的样例总数
    'stream_length': 100000,
    
    # 生成的数据集中，正负样本比例
    'imbalance_rate': [{0: 1, 1: 1}, {0: 10, 1: 1}, {0: 20, 1: 1}, {0: 50, 1: 1}, {0: 100, 1: 1}]
}
```

> scikit-multiflow 中的 [stream generator](https://scikit-multiflow.github.io/scikit-multiflow/documentation.html#stream-generators) 

运行 data_generator.py, 上面的 DataSets 会在 data/ 路径下自动生成若干个数据集。路径树如下：

+ data
    + noise-imbalance
        + 1
            + data.csv
            + data_info.csv
        + 2
            + data.csv
            + data_info.csv
        + 3
            + data.csv
            + data_info.csv
        + ...
        + 20
            + data.csv
            + data_info.csv
        + data_sets_info.csv

noise-imbalance 路径下有20个数据集（因为imbalance_rate有5个待测试的值，noise_percentage有4个待测试的值，排列组合一下5x4=20）  

这20个数据集依次从1编号，每个数据集中包含两个文件: data.csv，存放数据的文件，data_info.csv，存放数据集配置的文件
（只记录每个数据集不同的配置项，在这个例子中是imbalance_rate和noise_percentage这两个参数），如noise-imbalance/1下路径下的 data_info.csv 内容为：

```json
{
  "name": "noise-imbalance",
  "stream_generator": "HyperplaneGenerator",
  "stream_config": {
    "random_state": 666,
    "n_features": 4,
    "n_drift_features": 0,
    "mag_change": 0,
    "noise_percentage": 0,
    "sigma_percentage": 0
  },
  "stream_length": 100000,
  "imbalance_rate": {0: 1, 1: 1}
}
```

在 noise-imbalance 的根目录下还会**自动生成**一个 data_sets_info.csv 文件，里面描述了数据集0~19所共有的一些配置，如下所示：

```json
{
  "name": "noise-imbalance",
  "stream_generator": "HyperplaneGenerator",
  "stream_config": {
    "random_state": 666,
    "n_features": 4,
    "n_drift_features": 0,
    "mag_change": 0,
    "noise_percentage": [0, 0.01, 0.02, 0.05],
    "sigma_percentage": 0
  },
  "stream_length": 100000,
  "imbalance_rate": [{0: 1, 1: 1}, {0: 10, 1: 1}, {0: 20, 1: 1}, {0: 50, 1: 1}, {0: 100, 1: 1}],
  "test_params": ["stream_config:noise_percentage", "imbalance_rate"] # 该数据集集合测试的参数项列表（自动识别生成）
}
```

## 当使用 stream generator 为 ConceptDriftStream 时
ConceptDriftStream 比较特殊，该 stream generator 会将两个 stream 进行合并，在合并段会产生一次“概念漂移”，从而可以指定漂移的细节，详见 scikit-multiflow 文档
使用 ConceptDriftStream 的 data_generator.py 的一个配置例子：
```json
imbalance_drift_DataSets = {
    'name': 'imbalance-drift',
    'stream_generator': 'ConceptDriftStream',
    'stream_config': {
        'stream': {
            'name': 'noise-imbalance1',
            'stream_generator': 'HyperplaneGenerator',
            'stream_config': {
                'random_state': 666,
                'n_features': 4,
                'n_drift_features': 0,
                'mag_change': 0,
                'noise_percentage': [0, 0.01, 0.02, 0.05],
                'sigma_percentage': 0
            },
            'stream_length': 10000,
            'imbalance_rate': [{0: 1, 1: 1}, {0: 10, 1: 1}, {0: 20, 1: 1}, {0: 50, 1: 1}, {0: 100, 1: 1}]
        },
        'drift_stream': {
            'name': 'noise-imbalance2',
            'stream_generator': 'HyperplaneGenerator',
            'stream_config': {
                'random_state': 888,
                'n_features': 4,
                'n_drift_features': 0,
                'mag_change': 0,
                'noise_percentage': [0, 0.01, 0.02, 0.05],
                'sigma_percentage': 0
            },
            'stream_length': 10000,
            'imbalance_rate': [{0: 1, 1: 1}, {0: 10, 1: 1}, {0: 20, 1: 1}, {0: 50, 1: 1}, {0: 100, 1: 1}]
        },
        'position': 5000,
        'width': [1, 1000, 2000, 5000],
        'random_state': 666,
        'alpha': 0
    },
    'stream_length': 10000
}
```

## 实际数据集
实际数据集需整理成 data_generator.py 生成的虚拟数据集同样的格式，可参考“data_generator.py 使用说明”，具体要求如下

**data.csv**
+ 以逗号分隔开，每行存放一条数据，最后一列为标签 y

**data_info.json**
```json
{
  "name": "KDD99-version1", # 该数据集的名字
  "stream_length": 100000, # 该数据集的样本数量
  "imbalance_rate": {0:10, 1:11}, # 不平衡度
  #以上三条为必须项，根据实际情况可添加额外项，
  "test_params": ["imbalance_rate"] # 需要测试的参数项，可自行添加
}
```

**data_sets_info.json**
```json
{
  "name": "KDD99", # 该数据集集合的名字
  #以上为必须项，可根据需求添加额外项，所填加的项目会显示在对照试验的非对照参数说明中
  "test_params": ["imbalance_rate"] # 需要测试的参数项，可自行添加
}
```

# test.py 使用说明

设置工具开头的 data_set_paths, 运行该实验脚本，实验结果会自动写入 test\log.md 中。
如，想要对 data 下的 nois-imbalance 数据集集合进行实验，可以将 data_set_paths 设置为：
```python
data_sets_paths = ["..\\data\\concept-drift-with-noisy-imbalance"] # 添加额外的列表项可以一次进行多个实验
```

# model 实验说明
model 的实现必须满足一下接口

`parial_fit partial_fit(self, X, y, classes=None)`：流式处理中的增量式训练方法

`predict_proba(self, X)`: 对输入 X 进行概率预测
