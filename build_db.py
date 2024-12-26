import argparse
import logging
import os
import pandas as pd
import sqlite3

# Set up logging
logging.basicConfig(level=logging.INFO, format="[BUILD_DB %(asctime)s] - %(message)s")

# Define constants
DEFAULT_DATA_DIR = "data"
DEFAULT_DATABASE_NAME = os.path.join(DEFAULT_DATA_DIR, "etraveli.db")
DEFAULT_ERRANDS_FILE = os.path.join(DEFAULT_DATA_DIR, "errands.parquet")
DEFAULT_ORDERS_FILE = os.path.join(DEFAULT_DATA_DIR, "orders.parquet")
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def base36_to_decimal(value: str) -> int:
    """
    Convert a base36 encoded string to decimal integer using plain Python.

    @param value: A base36 encoded string.
    @return: The decimal integer value.
    """
    try:
        return int(value, 36)
    except ValueError as e:
        raise ValueError(f"Invalid base36 value '{value}': {e}")


def process_errands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the errands dataframe to match the schema.

    @param df: The input errands dataframe.
    @return: The processed errands dataframe
    """
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df = df.rename(
        columns={
            "is_test_errand": "is_test_errand",
            "order_number": "order_id",
            "created": "created",
        }
    )
    df["order_id"] = df["order_id"].apply(base36_to_decimal)
    df["created"] = pd.to_datetime(df["created"], format=DATE_FORMAT)
    return df


def process_orders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the orders dataframe to match the schema.

    @param df: The input orders dataframe.
    @return: The processed orders dataframe.
    """
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    df = df.rename(
        columns={
            "is_changed": "is_changed",
            "is_canceled": "is_canceled",
            "order_created_at": "order_created_at",
        }
    )
    df["order_created_at"] = pd.to_datetime(df["order_created_at"], format=DATE_FORMAT)
    return df


def update_orders_with_errand_counts(
    errands_df: pd.DataFrame, orders_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Update the orders dataframe with the count of errands for each order_id.

    @param errands_df: The processed errands dataframe.
    @param orders_df: The processed orders dataframe.
    @return: The updated orders dataframe.
    """
    logging.info("Computing errand counts for each order_id using pandas...")

    # Count occurrences of each order_id in errands
    errand_counts = errands_df["order_id"].value_counts().reset_index()
    errand_counts.columns = ["order_id", "count_errands"]

    # Merge counts into orders
    logging.info("Merging errand counts into orders dataframe...")
    orders_df = orders_df.merge(errand_counts, on="order_id", how="left")
    orders_df["count_errands"] = orders_df["count_errands"].fillna(0).astype(int)

    logging.info("Errand counts successfully added to orders dataframe.")
    return orders_df


def main(args: argparse.Namespace) -> None:
    """
    Entry point for the build_db.py script.

    @param args: The command-line arguments.
    """

    # Log the arguments
    logging.info(f"Using data directory: {args.data_dir}")
    logging.info(f"SQLite database name: {args.database_name}")
    logging.info(f"Errands file path: {args.errands_file}")
    logging.info(f"Orders file path: {args.orders_file}")
    logging.info(f"Subset mode: {'Enabled' if args.subset else 'Disabled'}")

    # Read parquet files
    logging.info("Reading Parquet files...")
    errands = pd.read_parquet(args.errands_file)
    orders = pd.read_parquet(args.orders_file)

    # Subset the data if requested
    if args.subset:
        logging.info("Processing only the top 100 rows for faster execution.")
        errands = errands.head(100)
        orders = orders.head(100)

    # Process dataframes
    logging.info("Processing errands dataframe...")
    errands = process_errands(errands)

    logging.info("Processing orders dataframe...")
    orders = process_orders(orders)

    # Update orders with count_errands
    logging.info("Updating orders with errand counts using pandas...")
    orders = update_orders_with_errand_counts(errands, orders)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.database_name), exist_ok=True)

    # Establish SQLite connection
    logging.info("Creating SQLite database and tables...")
    conn = sqlite3.connect(args.database_name)
    cursor = conn.cursor()

    # Create tables
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS errands (
        errand_id INTEGER PRIMARY KEY,
        is_test_errand BOOLEAN,
        created DATETIME,
        order_id INTEGER,
        FOREIGN KEY (order_id) REFERENCES orders(order_id)
    )
    """
    )
    cursor.execute(
        """
    CREATE INDEX IF NOT EXISTS idx_errands_order_id ON errands (order_id)
    """
    )
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY,
        is_changed BOOLEAN,
        is_canceled BOOLEAN,
        order_created_at DATETIME,
        count_errands INTEGER DEFAULT 0
    )
    """
    )
    cursor.execute(
        """
    CREATE INDEX IF NOT EXISTS idx_orders_is_changed ON orders (is_changed)
    """
    )
    cursor.execute(
        """
    CREATE INDEX IF NOT EXISTS idx_orders_is_canceled ON orders (is_canceled)
    """
    )

    # Write data to SQLite
    logging.info("Inserting data into errands table...")
    errands.to_sql("errands", conn, if_exists="replace", index=False)

    logging.info("Inserting updated orders data into orders table...")
    orders.to_sql("orders", conn, if_exists="replace", index=False)

    # Commit and close
    conn.commit()
    conn.close()
    logging.info("Database build completed successfully.")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Build an SQLite database from Parquet files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the data files.",
    )
    parser.add_argument(
        "--database_name",
        type=str,
        default=DEFAULT_DATABASE_NAME,
        help="Name of the SQLite database file.",
    )
    parser.add_argument(
        "--errands_file",
        type=str,
        default=DEFAULT_ERRANDS_FILE,
        help="Path to the errands Parquet file.",
    )
    parser.add_argument(
        "--orders_file",
        type=str,
        default=DEFAULT_ORDERS_FILE,
        help="Path to the orders Parquet file.",
    )
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Process only the top 100 rows of each file for testing.",
    )
    args = parser.parse_args()

    main(args)
