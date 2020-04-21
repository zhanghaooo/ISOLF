from models.Perceptron import Perceptron
from models.SGD import SGDClassifier
from models.CSOGD import CSOGDClassifier
from models.OFO import OFOClassifier
from models.OA3 import OA3Classifier
from models.OMCSL import OMCSL
from models.ACOG import ACOGClassifier
from models.OACSL import OACSL
from generator.data_set_generate import data_set_generate
from evaluator.evaluate import evaluate
import json


# generate data set
with open("..\\config\\imbalanced_budgeted_hyperplane_streams.json", 'r') as data_set_config_file:
    data_set_config = json.load(data_set_config_file)
data_set_generate(data_set_config, save_path="..\\data")

# measurement
measurement = 'F1_score'

# classical_models
classical_models = {
    'Perceptron': Perceptron(),
    'SGD': SGDClassifier(),
}

# models for imbalance data stream
imbalance_models = {
    # 'SGD': SGDClassifier(learning_rate='optimal'),
    'CSOGD-I': CSOGDClassifier(eta=0.1,
                               class_weight={-1: 0.1, 1: 0.9},
                               loss_func='I'),
    'CSOGD-II': CSOGDClassifier(eta=0.1,
                                class_weight={-1: 0.1, 1: 0.9},
                                loss_func='II'),
    'OFO': OFOClassifier(),
    'OMCSL': OMCSL(base_model=SGDClassifier(learning_rate='optimal'),
                   n_models=8,
                   measurement=measurement,
                   gamma=100),
    'ACOG-I': ACOGClassifier(eta=0.1,
                             class_weight={-1: 0.1, 1: 0.9},
                             gamma=1,
                             loss_func='I'),
    'ACOG-II': ACOGClassifier(eta=0.1,
                              class_weight={-1: 0.1, 1: 0.9},
                              gamma=1,
                              loss_func='II'),
    'OACSL': OACSL(base_model=SGDClassifier(learning_rate='optimal'),
                   window_size=10,
                   measurement=measurement,
                   eta_rho=0.1),
}

# models for drift imbalance data stream
drift_imbalance_models = {
    # 'SGD': SGDClassifier(learning_rate='optimal'),
    'CSOGD-I': CSOGDClassifier(eta=0.2,
                               class_weight={-1: 0.05, 1: 0.95},
                               loss_func='I'),
    'CSOGD-II': CSOGDClassifier(eta=0.2,
                                class_weight={-1: 0.05, 1: 0.95},
                                loss_func='II'),
    'OFO': OFOClassifier(),
    'OMCSL': OMCSL(base_model=SGDClassifier(learning_rate='optimal'),
                   n_models=8,
                   measurement=measurement,
                   gamma=100),
    'ACOG-I': ACOGClassifier(eta=0.1,
                             class_weight={-1: 0.1, 1: 0.9},
                             gamma=1,
                             loss_func='I'),
    'ACOG-II': ACOGClassifier(eta=0.1,
                              class_weight={-1: 0.1, 1: 0.9},
                              gamma=1,
                              loss_func='II'),
    'OACSL': OACSL(base_model=SGDClassifier(learning_rate='optimal'),
                   window_size=10,
                   measurement=measurement,
                   eta_rho=0.5),
}

# models for budget imbalance concept drift hyperplane streams
budget_imbalance_models = {
    'FCFQ': SGDClassifier(learning_rate='constant',
                          eta0=0.1,
                          class_weight={-1: 1, 1: 1},
                          query_strategy='FCFQ'),
    'RQ': SGDClassifier(learning_rate='constant',
                        eta0=0.1,
                        class_weight={-1: 1, 1: 1},
                        query_strategy='RQ',
                        query_ratio=1),
    'DPDQ': SGDClassifier(learning_rate='constant',
                          class_weight={-1: 1, 1: 1},
                          query_strategy='DPDQ',
                          query_threshold=4),
    'RPDQ': SGDClassifier(learning_rate='constant',
                          class_weight={-1: 1, 1: 1},
                          query_strategy='RPDQ',
                          query_weight=55),
    'AQ': SGDClassifier(learning_rate='constant',
                        class_weight={-1: 1, 1: 1},
                        query_strategy='AQ',
                        query_weight={-1: 40, 1: 60}),
    'BAAQ': SGDClassifier(learning_rate='constant',
                          class_weight={-1: 1, 1: 1},
                          query_strategy='BAAQ',
                          query_beta=1),
}

# models for imbalance budget concept drift hyperplane streams
imbalance_budget_models = {
    'FCFQ': SGDClassifier(learning_rate='constant',
                          eta0=0.1,
                          class_weight={-1: 1, 1: 1},
                          query_strategy='FCFQ'),
    'RQ': SGDClassifier(learning_rate='constant',
                        eta0=0.1,
                        class_weight={-1: 1, 1: 1},
                        query_strategy='RQ',
                        query_ratio=1),
    'DPDQ': SGDClassifier(learning_rate='constant',
                          class_weight={-1: 1, 1: 1},
                          query_strategy='DPDQ',
                          query_threshold=4),
    'RPDQ': SGDClassifier(learning_rate='constant',
                          class_weight={-1: 1, 1: 1},
                          query_strategy='RPDQ',
                          query_weight=55),
    'AQ': SGDClassifier(learning_rate='constant',
                        class_weight={-1: 1, 1: 1},
                        query_strategy='AQ',
                        query_weight={-1: 40, 1: 60}),
    'BAAQ': SGDClassifier(learning_rate='constant',
                          class_weight={-1: 1, 1: 1},
                          query_strategy='BAAQ',
                          query_beta=1),
    'OA3': OA3Classifier(eta=0.1,
                         class_weight={-1: 0.1, 1: 0.9},
                         query_weight={-1: 10, 1: 40},
                         gamma=1),
    'OACSL': OACSL(base_model=SGDClassifier(learning_rate='constant',
                                            eta0=0.1,
                                            query_strategy='BAAQ',
                                            query_beta=1),
                   measurement=measurement,
                   window_size=10,
                   eta_rho=0.1),
}

evaluate(models=imbalance_budget_models,
         data_set_path="..\\data\\ibcdhps",
         log_save_path="..\\log",
         measurement=measurement,
         test_times=1)
