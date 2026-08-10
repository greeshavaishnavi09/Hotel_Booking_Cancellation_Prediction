import os
import pandas as pd
from Hotel_Booking_Cancellation_Prediction.logging import logger
from Hotel_Booking_Cancellation_Prediction.entity.config_entity import DataValidationConfig

class DataValidation:

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_dataset(self):

        validation_status = True

        # 1. DATASET EXISTENCE CHECK

        if not os.path.exists(self.config.unzip_data_dir):
            logger.info("Dataset Not Found")
            return False

        logger.info("Dataset Found")

        # Read Dataset
        df = pd.read_csv(self.config.unzip_data_dir)

        # 2. DATASET SHAPE CHECK

        rows, columns = df.shape

        logger.info(f"Rows : {rows}")
        logger.info(f"Columns : {columns}")

        if rows == 0:
            logger.info("Dataset is Empty")
            validation_status = False

        if columns == 0:
            logger.info("Dataset has No Columns")
            validation_status = False

        # 3. SCHEMA / EXPECTED COLUMN CHECK

        expected_schema = self.config.all_schema

        expected_columns = list(expected_schema.keys())
        actual_columns = list(df.columns)

        missing_columns = set(expected_columns) - set(actual_columns)
        extra_columns = set(actual_columns) - set(expected_columns)

        if not missing_columns and not extra_columns:

            logger.info("Schema Validation Passed")

        else:

            logger.info("Schema Validation Failed")

            if missing_columns:
                logger.info(f"Missing Columns : {missing_columns}")

            if extra_columns:
                logger.info(f"Extra Columns : {extra_columns}")

            validation_status = False

        # 4. DATA TYPE VALIDATION

        actual_dtype = df.dtypes.astype(str).to_dict()

        for column, expected_dtype in expected_schema.items():

            if column not in actual_dtype:

                logger.info(f"{column} is Missing")
                validation_status = False

            elif actual_dtype[column] != expected_dtype:

                logger.info(
                    f"{column} datatype mismatch | "
                    f"Expected: {expected_dtype} | "
                    f"Actual: {actual_dtype[column]}"
                )

                validation_status = False

        logger.info("Data Type Validation Completed")

        # 5. MISSING VALUE VALIDATION

        missing_values = df.isnull().sum()

        logger.info("Missing Value Summary:")
        logger.info(missing_values[missing_values > 0])

        # These columns are known to contain legitimate missing values
        # in the original Hotel Booking dataset.
        allowed_missing_columns = [
            "children",
            "country",
            "agent",
            "company"
        ]

        unexpected_missing_columns = []

        for column in df.columns:

            if missing_values[column] > 0:

                if column not in allowed_missing_columns:
                    unexpected_missing_columns.append(column)

        if unexpected_missing_columns:

            logger.info(
                f"Unexpected Missing Values Found In: "
                f"{unexpected_missing_columns}"
            )

            validation_status = False

        else:

            logger.info("Missing Value Validation Passed")

        # 6. DUPLICATE ROW CHECK

        duplicates = df.duplicated().sum()

        logger.info(f"Duplicate Rows : {duplicates}")

        # Duplicates are reported for later preprocessing.
        # They are not treated as a schema failure at this stage.
        if duplicates > 0:
            logger.info(
                "Duplicate rows found. "
                "These will be handled during data preprocessing."
            )
        else:
            logger.info("No Duplicate Rows Found")

        # 7. TARGET COLUMN VALIDATION

        target_column = "is_canceled"

        if target_column not in df.columns:

            logger.info("Target Column Missing")
            validation_status = False

        else:

            logger.info("Target Column Found")

            target_values = set(df[target_column].dropna().unique())

            if target_values.issubset({0, 1}):

                logger.info(
                    "Target Validation Passed: "
                    "is_canceled contains only 0 and 1"
                )

            else:

                logger.info(
                    f"Invalid Target Values Found: {target_values}"
                )

                validation_status = False

        # 8. HOTEL COLUMN VALIDATION

        valid_hotel_values = [
            "City Hotel",
            "Resort Hotel"
        ]

        if not df["hotel"].isin(valid_hotel_values).all():

            logger.info("Invalid values found in hotel")
            validation_status = False

        else:

            logger.info("Hotel Category Validation Passed")

        # 9. MEAL COLUMN VALIDATION

        valid_meal_values = [
            "BB",
            "HB",
            "SC",
            "FB",
            "Undefined"
        ]

        if not df["meal"].isin(valid_meal_values).all():

            logger.info("Invalid values found in meal")
            validation_status = False

        else:

            logger.info("Meal Category Validation Passed")

        # 10. DEPOSIT TYPE VALIDATION

        valid_deposit_values = [
            "No Deposit",
            "Refundable",
            "Non Refund"
        ]

        if not df["deposit_type"].isin(valid_deposit_values).all():

            logger.info("Invalid values found in deposit_type")
            validation_status = False

        else:

            logger.info("Deposit Type Validation Passed")

        # 11. CUSTOMER TYPE VALIDATION

        valid_customer_types = [
            "Contract",
            "Group",
            "Transient",
            "Transient-Party"
        ]

        if not df["customer_type"].isin(valid_customer_types).all():

            logger.info("Invalid values found in customer_type")
            validation_status = False

        else:

            logger.info("Customer Type Validation Passed")

        # 12. DISTRIBUTION CHANNEL VALIDATION

        valid_distribution_channels = [
            "Direct",
            "Corporate",
            "TA/TO",
            "Undefined",
            "GDS"
        ]

        if not df["distribution_channel"].isin(
            valid_distribution_channels
        ).all():

            logger.info("Invalid values found in distribution_channel")
            validation_status = False

        else:

            logger.info("Distribution Channel Validation Passed")

        # 13. RESERVATION STATUS VALIDATION

        valid_reservation_status = [
            "Check-Out",
            "Canceled",
            "No-Show"
        ]

        if not df["reservation_status"].isin(
            valid_reservation_status
        ).all():

            logger.info("Invalid values found in reservation_status")
            validation_status = False

        else:

            logger.info("Reservation Status Validation Passed")

        # 14. NUMERICAL NEGATIVE VALUE CHECK

        non_negative_columns = [
            "lead_time",
            "arrival_date_week_number",
            "arrival_date_day_of_month",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "children",
            "babies",
            "is_repeated_guest",
            "previous_cancellations",
            "previous_bookings_not_canceled",
            "booking_changes",
            "days_in_waiting_list",
            "required_car_parking_spaces",
            "total_of_special_requests"
        ]

        for column in non_negative_columns:

            if (df[column].dropna() < 0).any():

                logger.info(
                    f"{column} contains negative values"
                )

                validation_status = False

        logger.info("Non-Negative Value Validation Completed")

        # 15. BINARY COLUMN VALIDATION

        binary_columns = [
            "is_canceled",
            "is_repeated_guest"
        ]

        for column in binary_columns:

            if not df[column].dropna().isin([0, 1]).all():

                logger.info(
                    f"{column} contains values other than 0 and 1"
                )

                validation_status = False

        logger.info("Binary Column Validation Completed")

        # 16. ADR VALIDATION

        # ADR is Average Daily Rate.
        # We report negative values instead of immediately failing
        # the entire dataset because this requires business-level
        # investigation during preprocessing.

        if (df["adr"].dropna() < 0).any():

            logger.info(
                "Warning: Negative ADR values detected. "
                "These will be investigated during preprocessing."
            )

        else:

            logger.info("ADR Validation Passed")

        # 17. WRITE VALIDATION STATUS

        os.makedirs(
            os.path.dirname(self.config.STATUS_FILE),
            exist_ok=True
        )

        with open(self.config.STATUS_FILE, "w") as f:

            f.write(
                f"Validation Status : {validation_status}"
            )

        logger.info(
            f"Validation Status : {validation_status}"
        )

        return validation_status