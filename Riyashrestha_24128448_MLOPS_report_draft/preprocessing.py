import pandas as pd
from sqlalchemy import create_engine
import numpy as np
import re
import redis
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import mlflow
import os

MLFLOW_TRACKING_URI = "http://localhost:5000" 
PREPROCESSING_EXPERIMENT_NAME = "UsedCars_Preprocessing"

def log(msg=None, level="INFO", section=None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if level in ["HEADER", "SUBHEADER"] and section:
        if level == "HEADER":
            print(f"\n{'='*80}")
            print(f"[{timestamp}] ## {section.upper()} ##")
            print(f"{'='*80}")
        elif level == "SUBHEADER":
            print(f"\n{'-'*60}")
            print(f"[{timestamp}] -- {section} --")
            print(f"{'-'*60}")
    else:
        if msg is None:
            raise ValueError("Message is required for regular log entries")
        if section:
            print(f"[{timestamp}] [{section}] {level}: {msg}")
        else:
            print(f"[{timestamp}] {level}: {msg}")

def store_in_redis(key, df): 
    log(section=f"REDIS STORAGE - {key}", level="SUBHEADER")
    log(f"Storing DataFrame in Redis with key: {key}...", section="REDIS")

    try:
        redis_conn = redis.Redis(host="127.0.0.1", port=6379)
        redis_conn.ping()
        table = pa.Table.from_pandas(df)
        buf = pa.BufferOutputStream()
        pq.write_table(table, buf)
        redis_conn.set(key, buf.getvalue().to_pybytes())

        log(f"Successfully stored {key} in Redis", section="REDIS")
        log(f"DataFrame shape: {df.shape}", section="REDIS")
        log(f"DataFrame memory: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB", section="REDIS") 
    except redis.exceptions.ConnectionError as r_conn_err:
        log(f"Redis connection error during storage: {r_conn_err}", level="ERROR", section="REDIS")
        raise 
    except Exception as e:
        log(f"Error storing data in Redis: {e}", level="ERROR", section="REDIS")
        raise 

def preprocess_data():
    log(section="PREPROCESSING EXECUTION", level="HEADER") 

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(PREPROCESSING_EXPERIMENT_NAME)
    log(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}", section="MLFLOW_SETUP")
    log(f"MLflow experiment set to: {PREPROCESSING_EXPERIMENT_NAME}", section="MLFLOW_SETUP")

    run_name = f"Preprocessing_Run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        log(f"MLflow Run ID: {run.info.run_id}", section="MLFLOW_RUN")

        try:
            log(section="DATA LOADING (MySQL)", level="SUBHEADER")
            DB_CONFIG = {
                "db_username": "riya",
                "db_host": "localhost",
                "db_port": "3306",
                "db_database": "usedcar"
            }
            mlflow.log_params({k: v for k, v in DB_CONFIG.items() if "password" not in k})
            mlflow.log_param("source_system", "MySQL")


            actual_db_config_for_connection = {
                "username": "riya",
                "password": "riyastha12#",
                "host": "localhost",
                "port": "3306",
                "database": "usedcar"
            }
            DB_URL = "mysql+pymysql://{username}:{password}@{host}:{port}/{database}".format(**actual_db_config_for_connection)
            engine = create_engine(DB_URL)
            log("Fetching data from usedcars_obt table...", section="DATA LOADING (MySQL)")
            df = pd.read_sql('SELECT * FROM usedcars_obt', con=engine)
            log(f"Successfully loaded data. Initial shape: {df.shape}", section="DATA LOADING (MySQL)")


            mlflow.log_metric("initial_rows", df.shape[0])
            mlflow.log_metric("initial_columns", df.shape[1])
            if 'car_id' in df.columns: 
                df.drop(columns=['car_id'], inplace=True)
                log(f"'car_id' column dropped. Shape after dropping: {df.shape}", section="INITIAL CLEANING")
                mlflow.log_param("car_id_dropped", True)


            log(section="CORE PREPROCESSING STEPS", level="SUBHEADER")
            df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
            df['log_price'] = np.log1p(df['price'])

            def fix_brand_model(row):
                if 'Rover' in str(row['model']) and row['brand'] == 'Land':
                    row['brand'] = 'Land Rover'
                elif 'Martin' in str(row['model']) and row['brand'] == 'Aston':
                    row['brand'] = 'Aston Martin'
                return row
            df = df.apply(fix_brand_model, axis=1)

            df['brand'] = df['brand'].str.lower().str.strip()
            df['model'] = df['model'].str.lower().str.strip()
            df['model_year'] = df['model_year'].astype(int)
            df['milage'] = df['milage'].replace('[^0-9]', '', regex=True).replace('', np.nan).astype(float)

            df['fuel_type'] = df['fuel_type'].replace(['–', '', np.nan], 'Unknown')
            df.loc[df['brand'].str.contains("tesla|lucid", case=False), 'fuel_type'] = 'Electric'

            def simplify_trans(trans):
                if pd.isnull(trans) or trans == '–':
                    return 'Unknown'
                trans = trans.lower()
                if 'manual' in trans or 'm/t' in trans: return 'Manual'
                elif 'cvt' in trans: return 'CVT'
                elif 'automatic' in trans or 'a/t' in trans: return 'Automatic'
                return 'Automatic'
            df['transmission'] = df['transmission'].apply(simplify_trans)

            for col in ['ext_col', 'int_col']:
                df[col] = df[col].str.lower().str.replace('[^\w\s]', '', regex=True).fillna('unknown')

            df['accident'] = df['accident'].map({'None reported': 0, 'At least 1 accident or damage reported': 1}).fillna(0)
            df['clean_title'] = df['clean_title'].map({'Yes': 1}).fillna(0)
            df['car_age'] = datetime.now().year - df['model_year'] 
            mlflow.log_param("current_year_for_age_calc", datetime.now().year)


            def extract_hp(val): match = re.search(r'(\d+(?:\.\d+)?)HP', str(val)); return float(match.group(1)) if match else np.nan
            def extract_displacement(val): match = re.search(r'(\d+(?:\.\d+)?)L', str(val)); return float(match.group(1)) if match else np.nan
            def extract_cylinders(val): match = re.search(r'(\d+)\s*Cylinder', str(val)); return int(match.group(1)) if match else np.nan
            def extract_fuel_engine(val):
                if pd.isnull(val): return "Unknown"
                for fuel in ['Flex Fuel', 'Gasoline', 'Diesel', 'Electric', 'Hybrid']:
                    if fuel.lower() in str(val).lower(): return fuel
                return "Unknown"
            def is_turbo(val): return int(bool(re.search(r'turbo|supercharged|sc', str(val).lower())))

            df['engine_hp'] = df['engine'].apply(extract_hp)
            df['engine_displacement_L'] = df['engine'].apply(extract_displacement)
            df['num_cylinders'] = df['engine'].apply(extract_cylinders)
            df['engine_fuel_detail'] = df['engine'].apply(extract_fuel_engine)
            df['is_turbo_supercharged'] = df['engine'].apply(is_turbo)

            df['milage_per_year'] = df['milage'] / (df['car_age'] + 1e-6) 
            df['brand_model'] = df['brand'] + '_' + df['model']

            luxury_brands = ['bmw', 'mercedes-benz', 'audi', 'lexus', 'porsche', 'jaguar', 'land rover', 'tesla', 'aston martin']
            df['is_luxury'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)
            df['is_electric'] = df['fuel_type'].str.lower().str.contains('electric').astype(int)
            df['is_hybrid'] = df['fuel_type'].str.lower().str.contains('hybrid|plug-in').astype(int)
            common_colors = ['black', 'white', 'silver', 'gray', 'grey', 'blue', 'red']
            df['is_common_color'] = df['ext_col'].apply(lambda x: 1 if any(c in x for c in common_colors) else 0)

            log("Handling missing values...", section="IMPUTATION")
            initial_missing_values = df.isnull().sum()
            cols_with_missing_before = initial_missing_values[initial_missing_values > 0].index.tolist()
            if cols_with_missing_before:
                mlflow.log_param("cols_with_missing_before_imputation", ", ".join(cols_with_missing_before))

            cat_cols = df.select_dtypes(include=['object']).columns
            num_cols = df.select_dtypes(include=[np.number]).columns.drop(['price', 'log_price'], errors='ignore')

            for col in cat_cols: df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            for col in num_cols: df[col] = df[col].fillna(df[col].median())
            
            total_missing_after = df.isnull().sum().sum()
            mlflow.log_metric("total_missing_after_imputation", total_missing_after)
            log(f"Total missing values after imputation: {total_missing_after}", section="IMPUTATION")


            log("Applying encoding and scaling...", section="ENCODING_SCALING")
            onehot_cols = ['fuel_type', 'transmission', 'engine_fuel_detail']
            df = pd.get_dummies(df, columns=onehot_cols, prefix=onehot_cols, drop_first=True)

            freq_encode_cols = ['brand', 'model', 'ext_col', 'int_col', 'brand_model']
            for col in freq_encode_cols:
                freq = df[col].value_counts(normalize=True)
                df[col + '_freq'] = df[col].map(freq)
            
            cols_to_drop_after_encoding = freq_encode_cols + ['engine']
            df.drop(columns=cols_to_drop_after_encoding, inplace=True, errors='ignore')
            mlflow.log_param("dropped_cols_after_encoding", ", ".join(cols_to_drop_after_encoding))


            scale_cols = ['milage', 'car_age', 'engine_hp', 'engine_displacement_L', 'num_cylinders', 'milage_per_year']
            actual_scale_cols = [col for col in scale_cols if col in df.columns]
            if actual_scale_cols:
                scaler = StandardScaler()
                df[actual_scale_cols] = scaler.fit_transform(df[actual_scale_cols])
                mlflow.log_param("scaled_columns", ", ".join(actual_scale_cols))

                def cap_outliers(series): lower, upper = series.quantile([0.01, 0.99]); return series.clip(lower, upper)
                df[actual_scale_cols] = df[actual_scale_cols].apply(cap_outliers)
                mlflow.log_param("outlier_capping_percentiles", "0.01_0.99")
            else:
                log("No columns found for scaling.", section="ENCODING_SCALING", level="WARNING")


            df.preprocesseddata = df.copy()

            mlflow.log_metric("final_rows", df.preprocesseddata.shape[0])
            mlflow.log_metric("final_columns", df.preprocesseddata.shape[1])
            log(f"Final preprocessed data shape: {df.preprocesseddata.shape}", section="FINALIZATION")


            log("Saving preprocessed data as MLflow artifact...", section="ARTIFACTS")
            temp_csv_filename = "preprocessed_usedcars_output.csv"
            df.preprocesseddata.to_csv(temp_csv_filename, index=False)
            mlflow.log_artifact(temp_csv_filename, artifact_path="preprocessed_dataset") 
            os.remove(temp_csv_filename) 
            log(f"Artifact '{temp_csv_filename}' logged to MLflow run under 'preprocessed_dataset/' and removed locally.", section="ARTIFACTS")


            log(f"Preprocessed data head (first 5 rows):", section="FINALIZATION")
            print(df.preprocesseddata.head().to_string())


            store_in_redis("usedcars_preprocessed_data", df.preprocesseddata)
            mlflow.set_tag("redis_storage_key", "usedcars_preprocessed_data")
            mlflow.set_tag("status", "completed")
            log("Preprocessing run completed successfully.", section="MLFLOW_RUN")
            
            return df.preprocesseddata

        except Exception as e:
            log(f"Error during preprocessing: {e}", level="ERROR", section="MLFLOW_RUN")
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error_message", str(e))
            import traceback
            mlflow.log_text(traceback.format_exc(), "error_traceback.txt")
            raise 

if __name__ == "__main__":
    log(section="MAIN EXECUTION", level="HEADER")
    try:
        preprocessed_df_main = preprocess_data()
        if preprocessed_df_main is not None:
            log(f"Preprocessing finished. Final DataFrame shape: {preprocessed_df_main.shape}", section="MAIN EXECUTION")
        else:
            log("Preprocessing returned None.", section="MAIN EXECUTION", level="WARNING")
    except Exception as e:
        log(f"Unhandled exception in main execution: {e}", level="CRITICAL", section="MAIN EXECUTION")
        # No MLflow context here unless preprocess_data failed to start one.
        # If preprocess_data started a run and failed, it would have logged the error.