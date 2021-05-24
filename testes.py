from adaptive_xgboost import AdaptiveXGBoostClassifier
from adaptive_incremental_ensemble import AdaptiveXGBoostClassifier2
from adaptive_incremental import Adaptive2
from adaptive_xgboost_thread import Adaptive3
from adaptive_incremental2 import Adaptive4
from adaptive_incremental2_adwin import Adaptive5

from skmultiflow.data import ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream

# Adaptive XGBoost classifier parameters
n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.2     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 1000  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = False    # Enable/disable drift detection

par_boost = [512,1024,2048,4096,8192]
classificadores = []
labels = []

# for x in par_boost:
# ## autor replace
#     # meu incremental
# AXGBg = Adaptive4(learning_rate=0.5,
#                                 max_depth=max_depth,
#                                 max_window_size=max_window_size,
#                                 min_window_size=min_window_size)
AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
                                n_estimators=100,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                max_window_size=max_window_size,
                                min_window_size=min_window_size,
                                detect_drift=detect_drift)
labels.append('incremental 1')
classificadores.append(AXGBr)

# AXGBg2 = Adaptive5(learning_rate=learning_rate,
#                                 max_depth=max_depth,
#                                 max_window_size=max_window_size,
#                                 min_window_size=min_window_size)
# classificadores.append(AXGBg)
# labels.append('incremental')

stream = FileStream("./datasets/airlines.csv")
# stream.prepare_for_use()   # Required for skmultiflow v0.4.1

# print(labels)
evaluator = EvaluatePrequential(pretrain_size=100000,
                                max_samples=500000,
                                show_plot=False,
                                metrics=["accuracy","running_time"])

evaluator.evaluate(stream=stream,
                   model=classificadores,
                   model_names=labels)
