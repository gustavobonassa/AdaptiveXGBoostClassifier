from adaptive_xgboost import AdaptiveXGBoostClassifier
from adaptive_incremental_ensemble import AdaptiveXGBoostClassifier2
from adaptive_incremental import Adaptive2
from adaptive_xgboost_thread import Adaptive3
from adaptive_incremental2 import Adaptive4

from skmultiflow.data import ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream

# Adaptive XGBoost classifier parameters
n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.3     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 1000  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = False    # Enable/disable drift detection

## autor push
# AXGBp = AdaptiveXGBoostClassifier(update_strategy='push',
#                                   n_estimators=n_estimators,
#                                   learning_rate=learning_rate,
#                                   max_depth=max_depth,
#                                   max_window_size=max_window_size,
#                                   min_window_size=min_window_size,
#                                   detect_drift=detect_drift)
## meu ensemble incremental
AXGBp2 = AdaptiveXGBoostClassifier2(update_strategy='push',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift)
## autor replace
AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift)

## meu incremental
AXGBg = Adaptive4(learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size)

## meu thread
AXGBt = Adaptive3(n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift)

# stream = ConceptDriftStream(random_state=1000,
#                             position=10000)
stream = FileStream("./datasets/sea_a.csv")
# stream.prepare_for_use()   # Required for skmultiflow v0.4.1


# classifier = SGDClassifier()
# classifier2 = KNNADWINClassifier(n_neighbors=8, max_window_size=2000,leaf_size=40, nominal_attributes=None)
# classifier3 = OzaBaggingADWINClassifier(base_estimator=KNNClassifier(n_neighbors=8, max_window_size=2000,
#                                         leaf_size=30))
# classifier4 = PassiveAggressiveClassifier()
# classifier5 = SGDRegressor()
# classifier6 = PerceptronMask()


evaluator = EvaluatePrequential(pretrain_size=0,
                                max_samples=100000,
                                show_plot=True,
                                metrics=["accuracy","running_time"])

evaluator.evaluate(stream=stream,
                   model=[AXGBr, AXGBt, AXGBg],
                   model_names=['AXGB ensemble', 'AXGB thread', 'AXGB incremental'])
