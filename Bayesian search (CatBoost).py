# CatBoost optimized using Bayesian search
import optuna
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def objective_cat(trial):
    params_cat = {
        'objective': 'Logloss',  
        'eval_metric': 'AUC',    
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 1, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
        'random_strength': trial.suggest_loguniform('random_strength', 1e-9, 10.0),
        'border_count': trial.suggest_int('border_count', 1, 255),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-9, 100),
        'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 0.01, 1.0),
        'boosting_type': trial.suggest_categorical('boosting_type', ['Ordered', 'Plain']),
        'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS', 'No'])
    }

    if params_cat['bootstrap_type'] == 'Bayesian':
        params_cat['bagging_temperature'] = trial.suggest_loguniform('bagging_temperature', 1e-2, 10.0)
    elif params_cat['bootstrap_type'] == 'Bernoulli':
        params_cat['subsample'] = trial.suggest_uniform('subsample', 0.5, 1.0)

    cat_model = CatBoostClassifier(**params_cat, silent=True)

    cv_scores = cross_val_score(cat_model, X_train, y_train, scoring='roc_auc')
    avg_cv_score = cv_scores.mean()

    return avg_cv_score

study_cat = optuna.create_study(direction='maximize')
study_cat.optimize(objective_cat, n_trials=100)

best_params_cat = study_cat.best_params
best_score_cat = study_cat.best_value

print("Best Parameters:", best_params_cat)
print("Best Score:", best_score_cat)

best_model_cat = CatBoostClassifier(**best_params_cat, silent=True)
best_model_cat.fit(X_train, y_train)
