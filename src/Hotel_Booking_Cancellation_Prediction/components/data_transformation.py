import pandas as pd
import joblib

from pathlib import Path
from Hotel_Booking_Cancellation_Prediction.logging import logger

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from Hotel_Booking_Cancellation_Prediction.entity.config_entity import DataTransformationConfig

class DataTransformation:

    def __init__(self, config):

        self.config = config

    # 1. Load dataset

    def load_data(self):

        df = pd.read_csv(self.config.data_path)

        return df

    # 2. Feature engineering + remove leakage columns

    def prepare_data(self, df):

        # Remove columns that contain information about
        # the final reservation outcome.
        columns_to_drop = [
            "reservation_status",
            "reservation_status_date"
        ]

        df = df.drop(
            columns=columns_to_drop,
            errors="ignore"
        )

        # Feature Engineering

        df["total_nights"] = (
            df["stays_in_weekend_nights"]
            + df["stays_in_week_nights"]
        )

        df["total_guests"] = (
            df["adults"]
            + df["children"].fillna(0)
            + df["babies"]
        )

        return df

    # 3. Separate features and target

    def separate_features_target(self, df):

        X = df.drop(
            columns=["is_canceled"]
        )

        y = df["is_canceled"]

        return X, y

    # 4. Identify numerical and categorical columns

    def get_feature_types(self, X):

        # These columns contain category/ID information
        # even though pandas may store them as numbers.
        categorical_columns = [
            "hotel",
            "arrival_date_month",
            "meal",
            "country",
            "market_segment",
            "distribution_channel",
            "reserved_room_type",
            "assigned_room_type",
            "deposit_type",
            "agent",
            "company",
            "customer_type"
        ]

        # Keep only columns that actually exist.
        categorical_columns = [
            col for col in categorical_columns
            if col in X.columns
        ]

        numerical_columns = [
            col for col in X.columns
            if col not in categorical_columns
        ]

        return numerical_columns, categorical_columns

    # 5. Create preprocessing pipelines

    def create_preprocessors(
        self,
        numerical_columns,
        categorical_columns
    ):

        # Numerical preprocessing for models
        # that require/benefit from scaling.
        numerical_pipeline_scaled = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        # Numerical preprocessing for tree-based models.
        # No scaling is required.
        numerical_pipeline_unscaled = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]
        )

        # Categorical preprocessing.
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        # Preprocessor for:
        # Logistic Regression / KNN / SVM
        preprocessor_scaled = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numerical_pipeline_scaled,
                    numerical_columns
                ),
                (
                    "cat",
                    categorical_pipeline,
                    categorical_columns
                )
            ]
        )

        # Preprocessor for:
        # Decision Tree / Random Forest
        preprocessor_unscaled = ColumnTransformer(
            transformers=[
                (
                    "num",
                    numerical_pipeline_unscaled,
                    numerical_columns
                ),
                (
                    "cat",
                    categorical_pipeline,
                    categorical_columns
                )
            ]
        )

        return (
            preprocessor_scaled,
            preprocessor_unscaled
        )

    # 6. Main transformation method

    def transform_data(self):

        # Load
        df = self.load_data()

        # Prepare
        df = self.prepare_data(df)

        # Separate X and y
        X, y = self.separate_features_target(df)

        # Identify feature types
        (
            numerical_columns,
            categorical_columns
        ) = self.get_feature_types(X)

        # Train/Test split BEFORE fitting preprocessing
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # Create preprocessors
        (
            preprocessor_scaled,
            preprocessor_unscaled
        ) = self.create_preprocessors(
            numerical_columns,
            categorical_columns
        )

        # Fit only on training data
        X_train_scaled = preprocessor_scaled.fit_transform(
            X_train
        )

        X_test_scaled = preprocessor_scaled.transform(
            X_test
        )

        X_train_unscaled = preprocessor_unscaled.fit_transform(
            X_train
        )

        X_test_unscaled = preprocessor_unscaled.transform(
            X_test
        )

        # Get generated feature names
        feature_names = (
            preprocessor_scaled
            .get_feature_names_out()
        )

        # Convert transformed arrays to DataFrames
        train_scaled_df = pd.DataFrame(
            X_train_scaled,
            columns=feature_names,
            index=X_train.index
        )

        test_scaled_df = pd.DataFrame(
            X_test_scaled,
            columns=feature_names,
            index=X_test.index
        )

        train_unscaled_df = pd.DataFrame(
            X_train_unscaled,
            columns=feature_names,
            index=X_train.index
        )

        test_unscaled_df = pd.DataFrame(
            X_test_unscaled,
            columns=feature_names,
            index=X_test.index
        )

        # Add target
        train_scaled_df["is_canceled"] = y_train.values
        test_scaled_df["is_canceled"] = y_test.values

        train_unscaled_df["is_canceled"] = y_train.values
        test_unscaled_df["is_canceled"] = y_test.values

        # Save scaled data
        train_scaled_df.to_csv(
            self.config.transformed_train_path,
            index=False
        )

        test_scaled_df.to_csv(
            self.config.transformed_test_path,
            index=False
        )

        # Save unscaled data

        train_unscaled_df.to_csv(
            self.config.transformed_unscaled_train_path,
            index=False
        )

        test_unscaled_df.to_csv(
            self.config.transformed_unscaled_test_path,
            index=False
        )

        # Save both preprocessors
        scaled_path = (
            self.config.root_dir
            / "preprocessor_scaled.pkl"
        )

        unscaled_path = (
            self.config.root_dir
            / "preprocessor_unscaled.pkl"
        )

        joblib.dump(
            preprocessor_scaled,
            scaled_path
        )

        joblib.dump(
            preprocessor_unscaled,
            unscaled_path
        )

        print("Data Transformation Completed")
        print("Training Shape:", train_scaled_df.shape)
        print("Testing Shape:", test_scaled_df.shape)
        print("Numerical Features:", len(numerical_columns))
        print("Categorical Features:", len(categorical_columns))

        return (
            train_scaled_df,
            test_scaled_df,
            train_unscaled_df,
            test_unscaled_df
        )