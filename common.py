from datetime import datetime
from IPython.display import Markdown, display
from pathlib import Path
import os
import pandas as pd
import sqlite3


def get_weekday(datetime_str):
    """
    Extract the weekday from a datetime string.

    @param datetime_str: A string representing a datetime in the format "YYYY-MM-DD HH:MM:SS".
    @return: The day of the week corresponding to the datetime (e.g., "Monday").
    """
    dt = pd.to_datetime(datetime_str)
    return dt.day_name()


def get_time_slot(datetime_str):
    """
    Determine the time slot for a given datetime string.

    Time slots are defined as:
      - "A" for 00:00 to 06:00
      - "B" for 06:01 to 12:00
      - "C" for 12:01 to 18:00
      - "D" for 18:01 to 23:59

    @param datetime_str: A string representing a datetime in the format "YYYY-MM-DD HH:MM:SS".
    @return: The time slot corresponding to the datetime.
    """
    dt = pd.to_datetime(datetime_str)
    hour = dt.hour
    if 0 <= hour <= 6:
        return "A"
    elif 6 < hour <= 12:
        return "B"
    elif 12 < hour <= 18:
        return "C"
    else:
        return "D"


def truncate_label(label: str | tuple, max_length: int) -> str:
    """
    Truncate a label or tuple to fit within the specified maximum length.

    @param label: The label to truncate.
    @param max_length: The maximum allowed length for the label.
    @return: A truncated label with an ellipsis if necessary.
    """
    if isinstance(label, tuple):
        half_length = max_length // 2
        return "/".join(
            [
                elem if len(elem) <= half_length else elem[: half_length - 3] + "…"
                for elem in label
            ]
        )
    return label if len(label) <= max_length else label[: max_length - 3] + "…"


def obtain_root_path() -> Path:
    """
    Obtain the path to the current directory.
    """

    return Path(os.getcwd())


def load_data(database_name, limits=None):
    """
    Load the SQLite database and return errands and orders as Pandas DataFrames.
    Optionally limit the number of rows loaded for each table.

    Parameters:
        database_name (str): Path to the SQLite database file.
        limits (tuple): A tuple of two integers (limit_errands, limit_orders) to limit rows. Defaults to None.

    Returns:
        tuple: errands and orders as Pandas DataFrames.
    """

    # Helper function to map SQLite types to Pandas dtypes
    def map_sqlite_to_pandas(sqlite_type):
        sqlite_type = sqlite_type.upper()
        if sqlite_type in ("TEXT", "VARCHAR", "CHAR"):
            return str
        elif sqlite_type in ("INTEGER", "INT"):
            return int
        elif sqlite_type in ("REAL", "FLOAT", "DOUBLE"):
            return float
        elif sqlite_type in ("TIMESTAMP", "DATETIME"):
            return "datetime64[ns]"
        else:
            return object

    # Establish SQLite connection
    conn = sqlite3.connect(database_name)

    # Fetch table info and define dtypes for errands and orders
    def get_dtypes(table_name):
        dtypes = {}
        query = f"PRAGMA table_info({table_name})"
        columns_info = conn.execute(query).fetchall()
        for col in columns_info:
            col_name = col[1]  # Column name
            col_type = col[2]  # Column type
            dtypes[col_name] = map_sqlite_to_pandas(col_type)
        return dtypes

    errands_dtypes = get_dtypes("errands")
    orders_dtypes = get_dtypes("orders")

    # Construct SQL queries with optional limits
    query_errands = "SELECT * FROM errands"
    query_orders = "SELECT * FROM orders"

    if limits:
        limit_errands, limit_orders = limits
        if limit_errands:
            query_errands += f" LIMIT {limit_errands}"
        if limit_orders:
            query_orders += f" LIMIT {limit_orders}"

    # Load data into Pandas DataFrames with specified dtypes
    errands = pd.read_sql_query(query_errands, conn)
    orders = pd.read_sql_query(query_orders, conn)

    # Convert columns to appropriate dtypes
    errands = errands.astype(errands_dtypes)
    orders = orders.astype(orders_dtypes)

    # Handle datetime conversion explicitly for TIMESTAMP columns
    for col, dtype in errands_dtypes.items():
        if dtype == "datetime64[ns]":
            errands[col] = pd.to_datetime(errands[col])

    for col, dtype in orders_dtypes.items():
        if dtype == "datetime64[ns]":
            orders[col] = pd.to_datetime(orders[col])

    # Close connection
    conn.close()

    return errands, orders


def render_markdown(markdown_content: str | list[str]) -> None:
    """
    Display markdown content in Jupyter Notebook.

    @param markdown_content: Markdown string or list of strings to display.
    @return: None.
    """
    if isinstance(markdown_content, list):
        markdown_content = "\n".join(markdown_content)
    display(Markdown(markdown_content))


def display_errands_stats(stats, overall_series=None):
    """
    Display the results of compute_contacts_stats as a Markdown table with outlier detection
    and percentage vectors for the distribution of errands.

    Parameters:
        stats (dict): The output of compute_contacts_stats.
        overall_series (pd.Series): The complete series to calculate overall statistics, if needed.
    """
    # Check if it's overall stats or stratified stats
    if "overall" in stats:
        overall_stats = stats["overall"]
        markdown_table = (
            "### Overall Statistics\n\n"
            "| Metric             | Value           |\n"
            "|--------------------|-----------------|\n"
        )
        for metric, value in overall_stats.items():
            if metric != "series":  # Exclude the raw series
                markdown_table += f"| {metric.capitalize()} | {value} |\n"
    else:
        # Compute overall mean and standard deviation from the overall_series
        overall_mean = overall_series.mean()
        overall_std = overall_series.std()

        # Dynamically determine the threshold for "small" groups
        all_counts = [group_stats["count"] for group_stats in stats.values()]
        small_threshold = pd.Series(all_counts).quantile(0.25)

        markdown_table = "### Stratified Statistics\n\n"
        markdown_table += "| Group     | Count | Mean | StD | Min | 25% | 50% | 75% | Max | GlobOutlier | InGrpOutlier | Percentage Vector |\n"
        markdown_table += "|-----------|-------|------|-----|-----|-----|-----|-----|-----|-------------|--------------|-------------------|\n"

        for group, group_stats in stats.items():
            group_series = group_stats["series"]  # Raw series data for this group
            group_count = group_stats["count"]
            group_mean = group_stats["mean"]

            # Compute Z-score for the group mean relative to the overall series
            z_score = (
                (group_mean - overall_mean) / overall_std if overall_std > 0 else 0
            )
            z_score_flag = (
                "*" if abs(z_score) > 2 and group_count > small_threshold else ""
            )

            # Compute mean-based Z-score for inter-group comparison
            other_means = [
                stats[other_group]["mean"]
                for other_group in stats
                if other_group != group
            ]
            other_mean = sum(other_means) / len(other_means)
            other_std = pd.Series(other_means).std()

            mean_z_score = (group_mean - other_mean) / other_std if other_std > 0 else 0
            mean_outlier_flag = (
                "*" if abs(mean_z_score) > 2 and group_count > small_threshold else ""
            )

            # Compute percentage vector for errands as ratios (0 to 1) and convert to native Python floats
            errand_counts = group_series.value_counts().sort_index()
            percentage_vector = [
                "%0.02f" % float(round(errand_counts.get(i, 0) / group_count, 2))
                for i in range(5)
            ]
            # Add 5+ as the last category
            percentage_vector.append(
                "%0.02f" % float(round(errand_counts[5:].sum() / group_count, 2))
            )

            # Convert the vector to a clean string representation
            percentage_vector_str = "[" + " ".join(percentage_vector) + "]"

            markdown_table += (
                f"| {group} | {group_stats['count']} | {group_stats['mean']:.2f} | "
                f"{group_stats['std_dev']:.2f} | {group_stats['min']} | "
                f"{group_stats['25th_percentile']} | {group_stats['median']} | "
                f"{group_stats['75th_percentile']} | {group_stats['max']} | {z_score_flag} | {mean_outlier_flag} | {percentage_vector_str} |\n"
            )

    # Display the Markdown
    render_markdown(markdown_table)
