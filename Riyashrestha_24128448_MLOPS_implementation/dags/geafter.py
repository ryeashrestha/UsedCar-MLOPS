import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
import pandas as pd
import redis
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO

def run_postprocessing_validation() -> bool:
    """
    Validate preprocessed dataset (loaded from Redis) using Great Expectations.
    :return: True if all validations pass, else False.
    """

    # --- Step 1: Load preprocessed data from Redis ---
    redis_key = "usedcars_preprocessed_data"
    redis_conn = redis.Redis(host="127.0.0.1", port=6379)

    print("\n===============================")
    print("Fetching preprocessed data from Redis...")
    buf = redis_conn.get(redis_key)
    if buf is None:
        print("No data found in Redis with key:", redis_key)
        return False

    reader = pa.BufferReader(buf)
    table = pq.read_table(reader)
    df = table.to_pandas()
    print(f"Loaded DataFrame from Redis: {df.shape}")

    # --- Step 2: Initialize Great Expectations context ---
    context = gx.get_context()

    # --- Step 3: Register a Pandas data source ---
    if "pandas_postprocess" not in [ds["name"] for ds in context.list_datasources()]:
        context.add_datasource(
            name="pandas_postprocess",
            class_name="Datasource",
            execution_engine={"class_name": "PandasExecutionEngine"},
            data_connectors={
                "runtime_data_connector": {
                    "class_name": "RuntimeDataConnector",
                    "batch_identifiers": ["default_identifier"]
                }
            }
        )

    # --- Step 4: Create or update expectation suite ---
    suite_name = "usedcar_postprocessed_suite"
    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    # --- Step 5: Create a batch request from DataFrame ---
    batch_request = RuntimeBatchRequest(
        datasource_name="pandas_postprocess",
        data_connector_name="runtime_data_connector",
        data_asset_name="usedcar_post_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier": "default_id"}
    )

    validator = context.get_validator(batch_request=batch_request, expectation_suite_name=suite_name)

    # --- Step 6: Define Expectations on Preprocessed Data ---

    validator.expect_column_values_to_be_between("milage", -5, 5)
    validator.expect_column_values_to_be_between("car_age", -5, 5)

    validator.expect_column_values_to_be_between("engine_hp", -5, 5)
    validator.expect_column_values_to_be_between("engine_displacement_L", -5, 5)
    validator.expect_column_values_to_be_between("num_cylinders", -5, 5)
    validator.expect_column_values_to_be_between("milage_per_year", -5, 5)

    validator.expect_column_values_to_be_in_set("is_luxury", [0, 1])
    validator.expect_column_values_to_be_in_set("is_electric", [0, 1])
    validator.expect_column_values_to_be_in_set("is_hybrid", [0, 1])
    validator.expect_column_values_to_be_in_set("is_common_color", [0, 1])
    validator.expect_column_values_to_be_in_set("is_turbo_supercharged", [0, 1])
    validator.expect_column_values_to_be_in_set("clean_title", [0, 1])
    validator.expect_column_values_to_be_in_set("accident", [0, 1])

    # --- Optional: Check for expected dummies (if known) ---
    expected_dummy_cols = [col for col in df.columns if "fuel_type_" in col or "transmission_" in col or "engine_fuel_detail_" in col]
    for col in expected_dummy_cols:
        validator.expect_column_to_exist(col)
        validator.expect_column_values_to_be_between(col, 0, 1)

    # --- Step 7: Save suite and validate ---
    validator.save_expectation_suite(discard_failed_expectations=False)
    results = validator.validate()

    # --- Step 8: Print validation results ---
    print("\nPOSTPROCESSING VALIDATION SUMMARY 📋")
    print("Success:", results.success)
    print("Stats:", results.statistics)

    for idx, result_item in enumerate(results.results, 1):
        exp = result_item["expectation_config"]["expectation_type"]
        success = result_item["success"]
        kwargs = result_item["expectation_config"]["kwargs"]

        print(f"\n🔹 Expectation {idx}: {exp}")
        print(f"   ➤ Column: {kwargs.get('column', 'N/A')}")
        print(f"   ➤ Success: {'Passed' if success else ' Failed'}")
        if not success:
            print(" Unexpected result:", result_item.get("result", {}))

    return results.success

# Run the validation
if __name__ == "__main__":
    validation_success = run_postprocessing_validation()
    if validation_success:
        print("\n All post-processing validations passed successfully!")
    else:
        print("\nSome validations failed. Please review the expectations.")
