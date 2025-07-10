import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from sqlalchemy import create_engine
import pandas as pd


def run_data_validation() -> bool:
    """
    Validate dataset from MariaDB using Great Expectations.
    :return: True if all validations pass, else False.
    """

    user = "riya"
    password = "riyastha12#"
    host = "localhost"    
    port = 3306              
    database = "usedcar"
    table_name = "usedcars_obt"

    engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")
    df = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)

    context = gx.get_context()

    if "pandas_datasource" not in [ds["name"] for ds in context.list_datasources()]:
        context.add_datasource(
            name="pandas_datasource",
            class_name="Datasource",
            execution_engine={"class_name": "PandasExecutionEngine"},
            data_connectors={
                "runtime_data_connector": {
                    "class_name": "RuntimeDataConnector",
                    "batch_identifiers": ["default_identifier"]
                }
            }
        )

    suite_name = "usedcar_expectation_suite"
    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    batch_request = RuntimeBatchRequest(
        datasource_name="pandas_datasource",
        data_connector_name="runtime_data_connector",
        data_asset_name="usedcar_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier": "default_id"}
    )

    validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)

    validator.expect_column_values_to_not_be_null("brand")
    validator.expect_column_values_to_be_between("model_year", min_value=1970, max_value=2025)

    if df["milage"].dtype == object:
        validator.expect_column_values_to_match_regex("milage", r"^\d{1,3}(,\d{3})*\smi\.$")

    validator.expect_column_values_to_not_be_null("price")
    if df["price"].dtype == object:
        validator.expect_column_values_to_match_regex("price", r"^\$\d{1,3}(,\d{3})*(\.\d{2})?$")

    validator.expect_column_distinct_values_to_be_in_set("accident", [
        "None reported", "At least 1 accident or damage reported"
    ])
    validator.expect_column_distinct_values_to_be_in_set("clean_title", ["Yes", None])

    validator.save_expectation_suite(discard_failed_expectations=False)

    result = validator.validate()
    print("Validation result:", result.success)
    return result.success
