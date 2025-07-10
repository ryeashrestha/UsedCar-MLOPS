import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import time

DB_CONFIG = {
    "username": "riya",
    "password": "riyastha12#",
    "host": "localhost",
    "port": "3306",
    "database": "usedcar"
}
DB_URL = "mysql+pymysql://{username}:{password}@{host}:{port}/{database}".format(**DB_CONFIG)
engine = create_engine(DB_URL)

def get_existing_column_values(table_name, column_name):
    query = f"SELECT DISTINCT {column_name} FROM {table_name}"
    try:
        return pd.read_sql(query, engine)[column_name].tolist()
    except Exception:
        return []

def get_existing_car_ids():
    try:
        return pd.read_sql("SELECT car_id FROM fact_used_cars", engine)["car_id"].tolist()
    except Exception:
        return []
        
def print_section_header(title):
    """Print a formatted major section header"""
    print("\n" + "="*80)
    print(f"{title.upper()}")
    print("="*80)
    
def print_subsection_header(title):
    """Print a formatted subsection header"""
    print("\n" + "-"*60)
    print(f"{title}")
    print("-"*60)
def print_status(message, status="INFO"):
    """Print a status message with appropriate prefix"""
    status_prefixes = {
        "INFO": "[INFO]     ",
        "SUCCESS": "[SUCCESS]  ",
        "WARNING": "[WARNING]  ",
        "ERROR": "[ERROR]    ",
        "STEP": "[STEP]     "
    }
    prefix = status_prefixes.get(status, "[INFO]     ")
    print(f"{prefix} {message}")

def get_condition_id(accident, clean_title):
    query = """
        SELECT condition_id
        FROM dim_condition
        WHERE accident = :accident AND clean_title = :clean_title
        LIMIT 1
    """
    with engine.connect() as conn:
        result = conn.execute(text(query), {"accident": accident, "clean_title": clean_title})
        row = result.fetchone()
        return row[0] if row else None
def ingest_data():
    start_time = time.time()
    print_section_header("USED CAR DATA INGESTION PROCESS")
    print_status(f"Process started at: {time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")

    # Loading the data
    print_subsection_header("DATA LOADING & PREPROCESSING")
    try:
        df = pd.read_csv('used_cars.csv')  
        original_count = len(df)
        print_status(f"Successfully loaded CSV with {original_count} rows", "SUCCESS")
    except Exception as e:
        print_status(f"Failed to load CSV: {e}", "ERROR")
        return

    print_status("Running data quality check...", "STEP")
    null_counts_before = df.isna().sum()
    total_nulls = null_counts_before.sum()

    categorical_cols = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 
                       'ext_col', 'int_col', 'accident', 'clean_title']
    numerical_cols = ['model_year', 'milage', 'price']

    if total_nulls > 0:
        print_status(f"Found {total_nulls} null values across {sum(null_counts_before > 0)} columns:", "WARNING")
        for col, count in null_counts_before[null_counts_before > 0].items():
            print(f"        - {col}: {count} nulls ({count/len(df)*100:.1f}%)")

        print_status("Imputing missing categorical values with mode...", "STEP")
        for col in categorical_cols:
            if col in df.columns and df[col].isna().any():
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
                print_status(f"Column '{col}': Filled {null_counts_before[col]} nulls with mode '{mode_value}'", "INFO")

        print_status("Imputing missing numerical values with median...", "STEP")
        for col in numerical_cols:
            if col in df.columns and df[col].isna().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                print_status(f"Column '{col}': Filled {null_counts_before[col]} nulls with median {median_value:.2f}", "INFO")

        null_counts_after = df.isna().sum()
        if null_counts_after.sum() == 0:
            print_status("All null values successfully imputed", "SUCCESS")
        else:
            print_status(f"Warning: {null_counts_after.sum()} null values remain", "WARNING")
    else:
        print_status("No null values found in the dataset", "SUCCESS")

    print_subsection_header("DIMENSION TABLES UPDATE")

    # Track statistics
    dim_stats = {
        'dim_brand': 0,
        'dim_model': 0,
        'dim_engine': 0,
        'dim_color': 0,
        'dim_condition': 0
    }

    print_status("Processing dim_brand table...", "STEP")
    existing_brands = get_existing_column_values('dim_brand', 'brand_name')
    if 'brand' in df.columns:
        new_brands = pd.DataFrame(df.loc[~df['brand'].isin(existing_brands), 'brand'].unique(), columns=['brand_name'])
        if not new_brands.empty:
            new_brands.to_sql('dim_brand', engine, if_exists='append', index=False, method='multi')
            dim_stats['dim_brand'] = len(new_brands)
            print_status(f"Added {len(new_brands)} new brands to dim_brand", "SUCCESS")
        else:
            print_status("No new brands to add", "INFO")
    else:
        print_status("Column 'brand' not found in dataset", "WARNING")

    print_status("Processing dim_model table...", "STEP")
    existing_models = get_existing_column_values('dim_model', 'model_name')
    if 'model' in df.columns:
        new_models = pd.DataFrame(df.loc[~df['model'].isin(existing_models), 'model'].unique(), columns=['model_name'])
        if not new_models.empty:
            new_models.to_sql('dim_model', engine, if_exists='append', index=False, method='multi')
            dim_stats['dim_model'] = len(new_models)
            print_status(f"Added {len(new_models)} new models to dim_model", "SUCCESS")
        else:
            print_status("No new models to add", "INFO")
    else:
        print_status("Column 'model' not found in dataset", "WARNING")

    print_status("Processing dim_engine table...", "STEP")
    existing_engines = get_existing_column_values('dim_engine', 'engine_type')
    if 'engine' in df.columns:
        new_engines = pd.DataFrame(df.loc[~df['engine'].isin(existing_engines), 'engine'].unique(), columns=['engine'])
        if not new_engines.empty:
            new_engines.to_sql('dim_engine', engine, if_exists='append', index=False, method='multi')
            dim_stats['dim_engine'] = len(new_engines)
            print_status(f"Added {len(new_engines)} new engine types to dim_engine", "SUCCESS")
        else:
            print_status("No new engine types to add", "INFO")
    else:
        print_status("Column 'engine' not found in dataset", "WARNING")

    print_status("Processing dim_color table...", "STEP")
    
    existing_ext_colors = get_existing_column_values('dim_color', 'ext_col')
    existing_int_colors = get_existing_column_values('dim_color', 'int_col')
    
    new_ext_colors = df['ext_col'].dropna().unique()
    new_int_colors = df['int_col'].dropna().unique()
    
    filtered_ext = [c for c in new_ext_colors if c not in existing_ext_colors]
    filtered_int = [c for c in new_int_colors if c not in existing_int_colors]
    
    color_records = []
    
    for color in filtered_ext:
        color_records.append({'ext_col': color, 'int_col': None})
    
    for color in filtered_int:
        color_records.append({'ext_col': None, 'int_col': color})

    new_colors_df = pd.DataFrame(color_records)
    
    # Insert into DB
    if not new_colors_df.empty:
        new_colors_df.to_sql('dim_color', engine, if_exists='append', index=False, method='multi')
        dim_stats['dim_color'] = len(new_colors_df)
        print_status(f"Added {len(new_colors_df)} new records to dim_color", "SUCCESS")
    else:
        print_status("No new colors to add", "INFO")

    print_status("Processing dim_condition table...", "STEP")
    
    if 'accident' in df.columns and 'clean_title' in df.columns:
        df['clean_title'] = df['clean_title'].map({'Yes': 1, 'No': 0})
    
        condition_df = df[['accident', 'clean_title']].dropna().drop_duplicates()
    
        existing_conditions = pd.read_sql('SELECT accident, clean_title FROM dim_condition', con=engine)
    
        new_conditions = pd.merge(condition_df, existing_conditions, how='outer', indicator=True)
        new_conditions = new_conditions[new_conditions['_merge'] == 'left_only'].drop(columns=['_merge'])
    
        if not new_conditions.empty:
            new_conditions.to_sql('dim_condition', engine, if_exists='append', index=False, method='multi')
            dim_stats['dim_condition'] = len(new_conditions)
            print_status(f"Added {len(new_conditions)} new rows to dim_condition", "SUCCESS")
        else:
            print_status("No new condition rows to add", "INFO")
    else:
        print_status("Columns 'accident' or 'clean_title' not found in dataset", "WARNING")

    id_column_map = {
        'dim_brand': 'brand_id',
        'dim_model': 'model_id',
        'dim_engine': 'engine_id',
        'dim_color': 'color_id',
        'dim_condition': 'condition_id'
    }

    def get_id(table, col, val):
        id_col = id_column_map[table]
        query = f"SELECT {id_col} FROM {table} WHERE {col} = :val LIMIT 1"
        with engine.connect() as conn:
            result = conn.execute(text(query), {"val": val})
            row = result.fetchone()
            return row[0] if row else None

    
    # ---------------- FACT TABLE ----------------
    print_subsection_header("FACT TABLE UPDATE")
    
    df['car_id'] = df.index.astype(str)
    
    existing_car_ids = get_existing_car_ids()
    df_fact = df[~df['car_id'].isin(existing_car_ids)]
    
    print_status(f"Total cars in CSV: {len(df)}", "INFO")
    print_status(f"Existing cars in database: {len(existing_car_ids)}", "INFO")
    print_status(f"New cars to process: {len(df_fact)}", "INFO")
    
    if not df_fact.empty:
        print_status("Processing new car entries...", "STEP")
        fact_rows = []
        skipped_rows = 0
        missing_keys = {
            'brand_id': 0,
            'model_id': 0,
            'engine_id': 0,
            'ext_color_id': 0,
            'int_color_id': 0,
            'condition_id': 0
        }

        for idx, row in df_fact.iterrows():
            if 'accident' in row and 'clean_title' in row:
                condition_desc = f"Accident: {row['accident']}, Clean Title: {row['clean_title']}"
            else:
                condition_desc = "Unknown"
    
            brand_id = get_id('dim_brand', 'brand_name', row.get('brand')) if 'brand' in row else None
            model_id = get_id('dim_model', 'model_name', row.get('model')) if 'model' in row else None
            engine_id = get_id('dim_engine', 'engine', row.get('engine')) if 'engine' in row else None
            ext_color_id = get_id('dim_color', 'ext_col', row.get('ext_col')) if 'ext_col' in row else None
            int_color_id = get_id('dim_color', 'int_col', row.get('int_col')) if 'int_col' in row else None

            condition_id = get_condition_id(row['accident'], row['clean_title'])  
    
            if brand_id is None: missing_keys['brand_id'] += 1
            if model_id is None: missing_keys['model_id'] += 1
            if engine_id is None: missing_keys['engine_id'] += 1
            if ext_color_id is None: missing_keys['ext_color_id'] += 1
            if int_color_id is None: missing_keys['int_color_id'] += 1
            if condition_id is None: missing_keys['condition_id'] += 1
    
            if None in [brand_id, model_id, condition_id]:
                skipped_rows += 1
                continue

            fact_rows.append({
                'car_id': row['car_id'],
                'model_year': row.get('model_year'),
                'milage': row.get('milage'),
                'fuel_type': row.get('fuel_type'),
                'transmission': row.get('transmission'),
                'price': row.get('price'),
                'brand_id': brand_id,
                'model_id': model_id,
                'engine_id': engine_id,
                'ext_color_id': ext_color_id,
                'int_color_id': int_color_id,
                'condition_id': condition_id
            })
    
            if len(df_fact) > 1000 and idx % 1000 == 0:
                print_status(f"Processed {idx} of {len(df_fact)} cars...", "INFO")
    
        if skipped_rows > 0:
            print_status(f"Skipped {skipped_rows} rows due to missing required foreign keys:", "WARNING")
            for key, count in missing_keys.items():
                if count > 0:
                    print(f"        - Missing {key}: {count} cars")
    
        if fact_rows:
            fact_df = pd.DataFrame(fact_rows)
            try:
                fact_df.to_sql('fact_used_cars', engine, if_exists='append', index=False, method='multi')
                print_status(f"Inserted {len(fact_df)} new cars into fact_used_cars", "SUCCESS")
            except SQLAlchemyError as e:
                print_status(f"Error inserting fact rows: {e}", "ERROR")
        else:
            print_status("No cars to insert after filtering", "WARNING")
    
    else:
        print_status("No new cars to insert", "INFO")


    # ---------------- OBT TABLE ----------------
    print_subsection_header("OBT TABLE UPDATE")
    try:
        print_status("Preparing OBT table update...", "STEP")
        df_obt = df
        df_obt.to_sql('usedcars_obt', engine, if_exists='replace', index=False, method='multi')
        print_status(f"Used Cars OBT table replaced with {len(df_obt)} rows", "SUCCESS")
    except SQLAlchemyError as e:
        print_status(f"Error updating OBT table: {e}", "ERROR")

    print_subsection_header("DATABASE SUMMARY")
    try:
        with engine.connect() as conn:
            tables = ['dim_brand', 'dim_model', 'dim_engine', 'dim_color', 'dim_condition', 'fact_used_cars', 'usedcars_obt']
            print_status("Current table row counts:", "INFO")

            for table in tables:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                print(f"        - {table}: {count:,} rows")
    except SQLAlchemyError as e:
        print_status(f"Error checking table counts: {e}", "ERROR")

    print_status("Dimension tables updated with:", "INFO")
    for dim, count in dim_stats.items():
        print(f"        - {dim}: {count} new values added")

    elapsed_time = time.time() - start_time
    print_section_header("PROCESS COMPLETE")
    print_status(f"Total execution time: {elapsed_time:.2f} seconds", "INFO")
    print_status(f"Process completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}", "SUCCESS")

if __name__ == "__main__":
    ingest_data()

    
