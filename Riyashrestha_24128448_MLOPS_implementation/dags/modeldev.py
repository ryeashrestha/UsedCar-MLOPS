import redis
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
import logging
import os 
import mlflow
import mlflow.xgboost 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_from_redis(key):
    try:
        logger.info("Connecting to Redis...")
        redis_conn = redis.Redis(host="127.0.0.1", port=6379)
        redis_conn.ping()
        logger.info(f"Connected to Redis successfully.")

        data = redis_conn.get(key)
        if data is None:
            raise ValueError(f"No data found for key: {key}")
        
        buf = pa.BufferReader(data)
        table = pq.read_table(buf)
        df = table.to_pandas()
        logger.info(f"Data loaded from Redis. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"[ERROR] {e}")
        return None


def regression_accuracy(y_true, y_pred, tolerance=None):
    if tolerance is None:
        median_y_true = np.median(y_true)
        if median_y_true == 0: 
            tolerance = 0.1 
        else:
            tolerance = 0.1 * median_y_true
    return np.mean(np.abs(y_pred - y_true) < tolerance)


def tune_hyperparameters(df):
    logger.info("Starting Optuna hyperparameter tuning...")

    y = df['price']
    y_log = np.log1p(y)
    X = df.drop(columns=['price', 'log_price'] if 'log_price' in df.columns else ['price'])

    X_train, X_val, y_train_log, y_val_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    dtrain = xgb.DMatrix(X_train, label=y_train_log)
    dval = xgb.DMatrix(X_val, label=y_val_log)

    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'lambda': trial.suggest_float('lambda', 0.1, 10.0),
            'alpha': trial.suggest_float('alpha', 0.1, 10.0),
            'seed': 30
        }
        model = xgb.train(param, dtrain, num_boost_round=1000,
                          evals=[(dval, 'eval')],
                          early_stopping_rounds=50,
                          verbose_eval=False)
        preds = model.predict(dval)
        return mean_squared_error(np.expm1(y_val_log), np.expm1(preds), squared=False)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    logger.info(f"Best trial: {study.best_trial.params}")
    return study.best_trial.params


def train_xgboost(df):
    mlflow.set_tracking_uri("http://localhost:5000") 
    mlflow.set_experiment("UsedCars_modeltraining")

    with mlflow.start_run() as run:
        run_id = run.info.run_id 
        logger.info(f"MLflow Run started with ID: {run_id}")
        mlflow.set_tag("mlflow.runName", f"xgboost_training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}")

        try:
            logger.info("Starting data preprocessing...")
            y = df['price']
            y_log = np.log1p(y)
            cols_to_drop = ['price']
            if 'log_price' in df.columns:
                cols_to_drop.append('log_price')
            X = df.drop(columns=cols_to_drop)

            X_train, X_test, y_train_log, y_test_log = train_test_split(
                X, y_log, test_size=0.2, random_state=42)

            dtrain = xgb.DMatrix(X_train, label=y_train_log, feature_names=X_train.columns.tolist())
            dtest = xgb.DMatrix(X_test, label=y_test_log, feature_names=X_test.columns.tolist())

            best_params = tune_hyperparameters(df)
            best_params.update({
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'seed': 30
            })

            mlflow.log_params(best_params)
            mlflow.log_param("train_test_split_random_state", 42)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("num_boost_round_config", 1000)
            mlflow.log_param("early_stopping_rounds_config", 20)

            logger.info("Training XGBoost model with early stopping...")
            model = xgb.train(best_params, dtrain, num_boost_round=1000,
                              evals=[(dtrain, 'train'), (dtest, 'eval')],
                              early_stopping_rounds=50, verbose_eval=False)

            mlflow.log_param("actual_num_boost_rounds", model.best_iteration + 1)

            logger.info("Serializing model for Redis...")
            model_bytes = bytes(model.save_raw())
            redis_conn = redis.Redis(host="127.0.0.1", port=6379)
            redis_conn.set("trained_xgboost_model", model_bytes)
            logger.info("Model saved to Redis under key 'trained_xgboost_model'.")

            y_train_pred_log = model.predict(dtrain)
            y_test_pred_log = model.predict(dtest)
            y_train_pred = np.expm1(y_train_pred_log)
            y_test_pred = np.expm1(y_test_pred_log)
            y_train = np.expm1(y_train_log)
            y_test = np.expm1(y_test_log)

            logger.info("Model trained. Calculating performance metrics...")
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_acc = regression_accuracy(y_train, y_train_pred)

            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("train_accuracy_10_percent_median", train_acc)
            logger.info(f"Train R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}, Accuracy: {train_acc*100:.2f}%")

            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_acc = regression_accuracy(y_test, y_test_pred)

            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("test_accuracy_10_percent_median", test_acc)
            logger.info(f"Test R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}, Accuracy: {test_acc*100:.2f}%")

            logger.info("Logging feature importances...")
            importance_dict = model.get_score(importance_type='weight')
            importance_df = pd.DataFrame({
                'feature': list(importance_dict.keys()),
                'importance': list(importance_dict.values())
            }).sort_values(by='importance', ascending=False)
            
            feature_importance_filename = "feature_importance.csv"
            importance_df.to_csv(feature_importance_filename, index=False)
            mlflow.log_artifact(feature_importance_filename, artifact_path="feature_analysis")
            os.remove(feature_importance_filename) 

            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path="xgboost-model", 
                input_example=X_train.head(), 
            )
            logger.info("XGBoost model logged to MLflow as run artifact.")

            registered_model_name = "XGBoostUsedCarsPriceModel" 
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/xgboost-model", 
                name=registered_model_name 
            )
            logger.info(f"Model registered in MLflow Model Registry as '{registered_model_name}'.")

            logger.info("Starting 5-Fold Cross-Validation...")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            r2_scores, rmse_scores, acc_scores = [], [], []

            mlflow.log_param("cv_n_splits", 5)
            mlflow.log_param("cv_random_state", 30)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                logger.info(f" Fold {fold+1}")
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

                dtrain_cv = xgb.DMatrix(X_tr, label=y_tr_log, feature_names=X_tr.columns.tolist())
                dval_cv = xgb.DMatrix(X_val, label=y_val_log, feature_names=X_val.columns.tolist())

                model_cv = xgb.train(best_params, dtrain_cv, num_boost_round=1000,
                                     evals=[(dtrain_cv, 'train'), (dval_cv, 'val')],
                                     early_stopping_rounds=50, verbose_eval=False)

                y_val_pred_log_cv = model_cv.predict(dval_cv)
                y_val_pred_cv = np.expm1(y_val_pred_log_cv)
                y_val_cv = np.expm1(y_val_log)

                r2_scores.append(r2_score(y_val_cv, y_val_pred_cv))
                rmse_scores.append(np.sqrt(mean_squared_error(y_val_cv, y_val_pred_cv)))
                acc_scores.append(regression_accuracy(y_val_cv, y_val_pred_cv))

            mean_cv_r2 = np.mean(r2_scores)
            mean_cv_rmse = np.mean(rmse_scores)
            mean_cv_acc = np.mean(acc_scores)

            mlflow.log_metric("cv_mean_r2", mean_cv_r2)
            mlflow.log_metric("cv_mean_rmse", mean_cv_rmse)
            mlflow.log_metric("cv_mean_accuracy_10_percent_median", mean_cv_acc)
            logger.info(f"CV Results: Mean R²: {mean_cv_r2:.4f}, Mean RMSE: ${mean_cv_rmse:,.2f}, Mean Acc: {mean_cv_acc*100:.2f}%")

            logger.info(f"MLflow Run completed successfully. Run ID: {run_id}")
            mlflow.set_tag("training_status", "success")

        except Exception as e:
            logger.error(f"[Training Error with MLflow] {e}", exc_info=True) 
            mlflow.set_tag("training_status", "failed")
            mlflow.set_tag("error_message", str(e))
            raise


def run_model_training():
    logger.info("Starting model training task...")
    df = load_from_redis("usedcars_preprocessed_data")
    if df is not None:
        train_xgboost(df)
    else:
        logger.warning("No data available. Training skipped.")
