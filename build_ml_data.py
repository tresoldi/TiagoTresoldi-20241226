import pandas as pd
import common

# global dictionaries
CURRENCIES = {
    "Euro": "EUR",
    "US Dollar": "USD",
    "Pound Sterling": "GBP",
    "Australian Dollar": "AUD",
    "Brazilian Real": "BRL",
    "Danish Krone": "DKK",
    "Saudi Riyal": "SAR",
    "Mexican Peso": "MXN",
    "Zloty": "PLN",
    "Norwegian Krone": "NOK",
    "Canadian Dollar": "CAD",
    "United Arab Emirates dirham": "AED",
    "Swedish Krona": "SEK",
    "Chilean Peso": "CLP",
    "Peso Uruguayo": "UYU",
    "Nuevo Sol Peru": "PEN",
    "South Korean Won": "KRW",
    "Malaysian Ringgit": "MYR",
    "Argentine Peso": "ARS",
    "Thai Baht": "THB",
    "Czech Koruna": "CZK",
    "Colombian Peso Colombia": "COP",
    "Kuwaiti Dinar": "KWD",
    "Swiss Franc": "CHF",
    "Hryvnia Ukraine": "UAH",
    "South African Rand": "ZAR",
    "Japanese yen": "JPY",
    "Jordanian Dinar": "JOD",
    "Bahraini Dinar": "BHD",
    "New Zealand Dollar": "NZD",
    "Indian Rupee": "INR",
    "Egyptian Pound": "EGP",
    "Bulgarian Lev": "BGN",
    "Rupiah Indonesia": "IDR",
    "Turkish Lira": "TRY",
    "Qatari Rial": "QAR",
    "Singapore Dollar": "SGD",
    "Hong Kong Dollar": "HKD",
    "Philippine Peso": "PHP",
    "New Taiwan Dollar": "TWD",
    "Rial Omani Oman": "OMR",
    "Forint": "HUF",
    "Yuan Renminbi": "CNY",
    "Vietnamese dong": "VND",
    "Iceland Krona": "ISK",
    "Tenge Kazakhstan": "KZT",
    "Uzbekistan Som": "UZS",
}
RATE = {
    "EUR": 0.95,
    "AUD": 1.45,
    "BRL": 5.10,
    "USD": 1.00,
    "DKK": 6.95,
    "SAR": 3.75,
    "MXN": 16.80,
    "GBP": 0.82,
    "PLN": 4.20,
    "NOK": 10.60,
    "CAD": 1.30,
    "AED": 3.67,
    "SEK": 10.50,
    "CLP": 930.00,
    "UYU": 40.50,
    "PEN": 3.80,
    "KRW": 1315.00,
    "MYR": 4.75,
    "ARS": 950.00,
    "THB": 36.00,
    "CZK": 23.10,
    "COP": 4000.00,
    "KWD": 0.31,
    "CHF": 0.87,
    "UAH": 39.00,
    "ZAR": 19.20,
    "JPY": 145.00,
    "JOD": 0.71,
    "BHD": 0.38,
    "NZD": 1.55,
    "INR": 84.00,
    "EGP": 31.50,
    "BGN": 1.80,
    "IDR": 15600,
    "TRY": 33.00,
    "QAR": 3.64,
    "SGD": 1.34,
    "HKD": 7.82,
    "PHP": 57.00,
    "TWD": 31.20,
    "OMR": 0.385,
    "HUF": 355.00,
    "CNY": 7.20,
    "VND": 24500,
    "ISK": 137.00,
    "KZT": 450.00,
    "UZS": 12400,
}


def transform_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the given dataframe according to the specified column modifications.

    @param df: The original Pandas dataframe to transform.
    @return: A new dataframe with transformed columns.
    """
    # Create a new dataframe with specified transformations
    new_df = pd.DataFrame()

    # Keep `order_id` as is
    new_df["order_id"] = df["order_id"]

    # Replace `pnr` with `pnr_size`
    new_df["pnr_size"] = df["pnr"].apply(lambda x: len(x.split(",")))

    # Process `booking_system`, `brand`, and `partner`
    for col, substring in {
        "booking_system": "System ",
        "brand": "Brand ",
        "partner": "Partner ",
    }.items():
        new_df[col] = df[col].apply(lambda x: x.replace(substring, "").strip())

    # Replace `currency` values using `CURRENCIES` and raise error for missing mappings
    new_df["currency"] = df["currency"].apply(
        lambda x: (
            CURRENCIES[x]
            if x in CURRENCIES
            else (_ for _ in ()).throw(
                ValueError(f"Currency '{x}' not mapped in CURRENCIES.")
            )
        )
    )

    # Convert `order_amount` to USD using `RATE` and raise error for missing rates
    new_df["order_amount"] = df.apply(
        lambda row: (
            row["order_amount"] * RATE[CURRENCIES[row["currency"]]]
            if CURRENCIES[row["currency"]] in RATE
            else (_ for _ in ()).throw(
                ValueError(
                    f"Rate for '{CURRENCIES[row['currency']]}' not found in RATE."
                )
            )
        ),
        axis=1,
    )
    new_df["order_amount"] = new_df["order_amount"].round(2)

    # Copy remaining specified columns as is
    columns_to_copy = [
        "customer_group_type",
        "device",
        "client_entry_type",
        "booking_system_source_type",
        "origin_country",
        "journey_type_id",
        "is_changed",
        "is_canceled",
        "cancel_reason",
        "change_reason",
        "count_errands",
        "order_created_at",
    ]
    for col in columns_to_copy:
        new_df[col] = df[col]

    # Decompose `order_created_at` into `weekday` and `time_slot`
    new_df["weekday"] = df["order_created_at"].apply(common.get_weekday)
    new_df["time_slot"] = df["order_created_at"].apply(common.get_time_slot)

    # Add `zero_errands` column
    new_df["zero_errands"] = new_df["count_errands"] == 0

    return new_df


def transform_errands(
    errands_df: pd.DataFrame, orders_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Transforms the errands_df using information from orders_df based on specified requirements.

    @param errands_df: The dataframe containing errands information.
    @param orders_df: The dataframe containing order information, with order_id as primary key.
    @return: The transformed errands_df dataframe.
    """
    # Ensure `order_id` is a string in both DataFrames
    errands_df["order_id"] = errands_df["order_id"].astype(str)
    orders_df["order_id"] = orders_df["order_id"].astype(str)

    # Step 1: Remove rows where `is_test_errand` is True
    errands_df = errands_df[~errands_df["is_test_errand"]]
    if errands_df["is_test_errand"].any():
        raise ValueError(
            "Rows with 'is_test_errand' set to True were not properly removed."
        )

    # Step 2: Merge errands_df with orders_df using order_id as the foreign key
    merged_df = errands_df.merge(
        orders_df, how="left", on="order_id", suffixes=("", "_order")
    )

    # Step 3: Calculate additional fields
    merged_df["weekday"] = merged_df["created"].apply(common.get_weekday)
    merged_df["time_slot"] = merged_df["created"].apply(common.get_time_slot)
    merged_df["order_diff"] = (
        pd.to_datetime(merged_df["created"])
        - pd.to_datetime(merged_df["order_created_at"])
    ).dt.total_seconds() / 3600

    # Step 4: Select and rename the required columns
    transformed_df = merged_df[
        [
            "errand_id",
            "order_id",
            "created",
            "weekday",
            "time_slot",
            "order_diff",
            "errand_category",
            "errand_type",
            "errand_action",
            "errand_channel",
            "pnr_size",
            "booking_system",
            "brand",
            "partner",
            "order_amount",
            "customer_group_type",
            "device",
            "client_entry_type",
            "origin_country",
            "journey_type_id",
        ]
    ]

    # Step 5: Sort by 'order_id' and 'order_diff'
    transformed_df = transformed_df.sort_values(by=["order_id", "order_diff"])

    # Step 6: Add 'errand_order' column
    transformed_df["errand_order"] = transformed_df.groupby("order_id").cumcount() + 1

    return transformed_df


#############################
# Load data
DB_FILE = common.obtain_root_path() / "data" / "etraveli.db"
LIMITS = None  # (28800, 63000)  # Use `None` to disable limits
errands_df, orders_df = common.load_data(DB_FILE, LIMITS)

# Column transformations
errands_df["is_test_errand"] = errands_df["is_test_errand"].astype(bool)
errands_df["order_id"] = errands_df["order_id"].astype(str)
orders_df["is_changed"] = orders_df["is_changed"].astype(bool)
orders_df["is_canceled"] = orders_df["is_canceled"].astype(bool)

# Build ML dataset
print("Transforming orders...")
transformed_orders_df = transform_orders(orders_df)
print("Done.\n")
print(transformed_orders_df.head())

print("Transforming errands...")
transformed_errands_df = transform_errands(errands_df, transformed_orders_df)
print("Done.\n")
print(transformed_errands_df.head())

# Save the DataFrame to Parquet
transformed_orders_df.to_parquet(
    common.obtain_root_path() / "data" / "orders_ml.parquet", index=False
)
transformed_errands_df.to_parquet(
    common.obtain_root_path() / "data" / "errands_ml.parquet", index=False
)
