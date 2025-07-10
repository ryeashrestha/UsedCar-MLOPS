from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python_operator import PythonOperator
from ingest import ingest_data
from datetime import datetime, timedelta
from preprocessing import preprocess_data
from gebefore import run_data_validation
from geafter import run_postprocessing_validation
from modeldev import run_model_training  

default_args = {
"owner": "Riya Shrestha",
"depends_on_past": False,
"email": ["Riya.Shrestha@mail.bcu.ac.uk"],
"email_on_failure": False,
"email_on_retry": False,
"retries": 0,
"retry_delay": timedelta(minutes=5)
}
with DAG(
    "UsedCarPrediction",
    default_args=default_args,
    description="""
        End-to-end podcast analytics and machine learning pipeline.
        Implements ingestion to a star schema in MariaDB, preprocessing and caching with Redis,
        model training and evaluation tracked via MLflow, and best model deployment using FastAPI.
        """,
    schedule_interval=timedelta(days=1),
    start_date=datetime(2021, 10, 10),
    catchup=False,
    tags=["example"],
) as dag:
    start_mcs_container = BashOperator(
        task_id='run_mcs_container',
        bash_command='docker start mcs_container',
        dag=dag,
    )
    start_redis_store = BashOperator(
        task_id='run_redis_store',
        bash_command='docker start redis_store',
        dag=dag,
    )
    mlflow_run = BashOperator(
    task_id='mlflow_run',
    bash_command="""
    if ! lsof -i:5000; then
        nohup conda run -n usedcar mlflow ui --host 0.0.0.0 --port 5000 > /tmp/mlflow_ui.log 2>&1 &
        echo "MLflow UI started"
    else
        echo "MLflow UI already running"
    fi
    """,
    dag=dag,
    )
    data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_data
    )
    ge_validation = PythonOperator(
        task_id="ge_predata_validation",
        python_callable=run_data_validation,
        dag=dag,
    )
    data_preprocessing = PythonOperator(
        task_id="data_preprocessing",
        python_callable=preprocess_data,
        dag=dag
    )
    postprocessing_validation = PythonOperator(
        task_id="ge_post-processing_validation",
        python_callable=run_postprocessing_validation,
        dag=dag
    )
    train_model= PythonOperator(
        task_id='train_XGB_model',
        python_callable=run_model_training,
        dag=dag
    )
    [start_mcs_container, start_redis_store] >> data_ingestion >> ge_validation >> data_preprocessing >> postprocessing_validation >> train_model 
