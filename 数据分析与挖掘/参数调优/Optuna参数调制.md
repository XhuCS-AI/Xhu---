### 一句话，这就是炼丹！

下面给出几个模型的常用炼丹参数：

1. XGBoost 回归模型：

```py
from xgboost import XGBRegressor
params = {
            'objective': trial.suggest_categorical('objective', ['reg:tweedie', 'reg:pseudohubererror']),
            'random_state': SEED,
            'num_parallel_tree': trial.suggest_int('num_parallel_tree', 2, 30),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.05, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e-1, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e-1, log=True),
        }
if params['objective'] == 'reg:tweedie':
    params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1, 2)
```

2. XGBoost 分类模型：

```py
from xgboost import XGBClassifier

params = {
    'objective': trial.suggest_categorical('objective', ['binary:logistic', 'multi:softmax']),  # 适合分类任务的目标函数
    'random_state': SEED,
    'num_parallel_tree': trial.suggest_int('num_parallel_tree', 2, 30),
    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
    'max_depth': trial.suggest_int('max_depth', 2, 4),
    'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.05, log=True),
    'subsample': trial.suggest_float('subsample', 0.5, 0.8),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
    'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e-1, log=True),
    'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1e-1, log=True),
}

# 如果使用多分类任务（multi:softmax），你可能需要设置 num_class（类别数）
if params['objective'] == 'multi:softmax':
    params['num_class'] = trial.suggest_int('num_class', 2, 10)  # 根据数据集的类别数调整
```

3. LightGBM 回归模型：

```py
from lightgbm import LGBMRegressor
params = {
            'objective': trial.suggest_categorical('objective', ['poisson', 'tweedie', 'regression']),
            'random_state': SEED,
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100)
        }
if params['objective'] == 'tweedie':
    params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1, 2)
```

4. LightGBM 分类模型：

```py
from lightgbm import LGBMClassifier
params = {
    'objective': trial.suggest_categorical('objective', ['binary', 'multiclass']),  # 分类任务的目标
    'random_state': SEED,
    'verbosity': -1,
    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
    'max_depth': trial.suggest_int('max_depth', 2, 4),
    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
    'subsample': trial.suggest_float('subsample', 0.5, 0.8),
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100)
}

# 如果使用多分类任务（multiclass），需要设置 num_class（类别数）
if params['objective'] == 'multiclass':
    params['num_class'] = trial.suggest_int('num_class', 2, 10)  # 根据数据集的类别数调整
```

5. CatBoost 回归模型：

```py
from catboost import CatBoostRegressor
params = {
            'loss_function': trial.suggest_categorical('objective', ['Tweedie:variance_power=1.5',
                                                                     'Poisson', 'RMSE']),
            'random_state': SEED,
            'iterations': trial.suggest_int('iterations', 100, 300),
            'depth': trial.suggest_int('depth', 2, 4),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 1e-1),
            'subsample': trial.suggest_float('subsample', 0.5, 0.7),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 60),
        }
```

6. CatBoost 分类模型：

```py
from catboost import CatBoostClassifier

params = {
    'loss_function': trial.suggest_categorical('loss_function', ['Logloss', 'MultiClass']),  # 分类任务的损失函数
    'random_state': SEED,
    'iterations': trial.suggest_int('iterations', 100, 300),
    'depth': trial.suggest_int('depth', 2, 4),
    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.05),
    'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 1e-1),
    'subsample': trial.suggest_float('subsample', 0.5, 0.7),
    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
    'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0),
    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 60),
}

# 如果是多分类任务，需要设置 classes_count 参数
if params['loss_function'] == 'MultiClass':
    params['classes_count'] = trial.suggest_int('classes_count', 2, 10)  # 根据数据集的类别数调整
```
