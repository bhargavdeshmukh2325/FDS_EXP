
import pandas as pd
import numpy as np

CSV_FILE = "Rajasthan_Heatwave_2006_2025.csv"
TARGET = "TMAX"

SIMPLE_FEATURES = ["TEMP2M"]
MULTIPLE_FEATURES = ["TEMP2M", "TMIN", "DEW2M", "SRAD"]


def main():
    print("EXPERIMENT 5: REGRESSION IMPUTATION")
    print("=" * 45)

    # Read dataset
    df = pd.read_csv(CSV_FILE)
    print("Dataset loaded successfully.")
    print("Rows and columns:", df.shape)

    print("\nDATASET INFORMATION")
    df.info()

    # Check required columns
    required = [TARGET] + MULTIPLE_FEATURES
    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        print("Required columns not found:", missing_columns)
        print("Available columns:", list(df.columns))
        return

    # Convert required columns to numeric
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nMISSING VALUES BEFORE PREDICTION")
    print(df[required].isnull().sum())

    # Use complete records to train and test the models
    complete = df.dropna(subset=required)

    if len(complete) < 10:
        print("\nNot enough complete records for regression.")
        return

    test = complete.sample(frac=0.2, random_state=42)
    train = complete.drop(test.index)

    # -----------------------------------------
    # 1. SIMPLE LINEAR REGRESSION
    # -----------------------------------------
    print("\n1. SIMPLE LINEAR REGRESSION")

    x = train["TEMP2M"].to_numpy(dtype=float)
    y = train["TMAX"].to_numpy(dtype=float)

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    denominator = np.sum((x - x_mean) ** 2)

    if denominator == 0:
        print("Cannot perform regression: TEMP2M is constant.")
        return

    slope = np.sum((x - x_mean) * (y - y_mean)) / denominator
    intercept = y_mean - slope * x_mean

    actual = test["TMAX"].to_numpy(dtype=float)
    x_test = test["TEMP2M"].to_numpy(dtype=float)

    simple_pred = intercept + slope * x_test

    print("Intercept:", round(intercept, 4))
    print("Slope:", round(slope, 4))
    print("Equation: TMAX =", round(intercept, 4),
          "+", round(slope, 4), "* TEMP2M")

    # -----------------------------------------
    # 2. MULTIPLE LINEAR REGRESSION
    # -----------------------------------------
    print("\n2. MULTIPLE LINEAR REGRESSION")

    X_train = train[MULTIPLE_FEATURES].to_numpy(dtype=float)
    X_test = test[MULTIPLE_FEATURES].to_numpy(dtype=float)
    y_train = train["TMAX"].to_numpy(dtype=float)

    # Add intercept column
    X_train = np.column_stack((np.ones(len(X_train)), X_train))
    X_test = np.column_stack((np.ones(len(X_test)), X_test))

    # Calculate coefficients using matrix algebra
    coefficients = np.linalg.pinv(X_train) @ y_train
    multiple_pred = X_test @ coefficients

    print("Coefficients:", np.round(coefficients, 4))
    print("Predictors:", MULTIPLE_FEATURES)

    # -----------------------------------------
    # 3. ACCURACY EVALUATION
    # -----------------------------------------
    def evaluate(actual, predicted):
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        total = np.sum((actual - np.mean(actual)) ** 2)
        error = np.sum((actual - predicted) ** 2)

        if total == 0:
            r2 = float("nan")
        else:
            r2 = 1 - error / total

        print("MAE:", round(mae, 4))
        print("RMSE:", round(rmse, 4))
        print("R-squared:", round(r2, 4))

    print("\nSimple Regression Accuracy")
    evaluate(actual, simple_pred)

    print("\nMultiple Regression Accuracy")
    evaluate(actual, multiple_pred)

    # Display sample predictions
    results = pd.DataFrame({
        "Actual_TMAX": actual,
        "Simple_Prediction": simple_pred,
        "Multiple_Prediction": multiple_pred
    })

    print("\nSAMPLE ACTUAL AND PREDICTED VALUES")
    print(results.head(10).round(2).to_string(index=False))

    # -----------------------------------------
    # 4. PREDICT MISSING TMAX VALUES
    # -----------------------------------------
    print("\n4. MISSING VALUE PREDICTION")

    original_missing = df[TARGET].isnull().sum()

    # Simple regression predictions
    simple_mask = df[TARGET].isnull() & df["TEMP2M"].notnull()

    df["TMAX_Predicted_Simple"] = np.nan
    df.loc[simple_mask, "TMAX_Predicted_Simple"] = (
        intercept + slope * df.loc[simple_mask, "TEMP2M"]
    )

    # Multiple regression predictions
    multiple_mask = (
        df[TARGET].isnull()
        & df[MULTIPLE_FEATURES].notnull().all(axis=1)
    )

    df["TMAX_Imputed"] = df[TARGET]

    if multiple_mask.any():
        X_missing = df.loc[
            multiple_mask, MULTIPLE_FEATURES
        ].to_numpy(dtype=float)

        X_missing = np.column_stack(
            (np.ones(len(X_missing)), X_missing)
        )

        df.loc[multiple_mask, "TMAX_Imputed"] = (
            X_missing @ coefficients
        )

    print("Missing TMAX before:", int(original_missing))
    print("Predictions by simple regression:", int(simple_mask.sum()))
    print("Values filled by multiple regression:", int(multiple_mask.sum()))
    print("Missing TMAX after imputation:",
          int(df["TMAX_Imputed"].isnull().sum()))

    # Save output file
    output_file = "rajasthan_exp5_imputed.csv"
    df.to_csv(output_file, index=False)
    print("\nSaved output file:", output_file)

    # -----------------------------------------
    # 5. MODEL COMPARISON
    # -----------------------------------------
    print("\n5. MODEL COMPARISON")
    print("Simple regression uses one predictor: TEMP2M.")
    print("Multiple regression uses four weather predictors.")

    print("\nCompare the errors:")
    print("Lower MAE and RMSE indicate smaller prediction errors.")
    print("Higher R-squared generally indicates a better fit.")

    print("\nEXPERIMENT 5 COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
