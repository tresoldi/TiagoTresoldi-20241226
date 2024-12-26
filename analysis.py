# Import standard libraries
from typing import List, Optional, Tuple, Dict
import logging

# Import third-party libraries
import pandas as pd
from tabulate import tabulate

# Import local modules
import common
import plotting


def display_numerical_summary(df: pd.DataFrame) -> None:
    """
    Display numerical summary of a DataFrame.

    @param df: The DataFrame to summarize.
    @return: None.
    """

    num_summary = df.describe().T

    report = ["#### Numerical Summary\n", num_summary.to_markdown(), "\n"]

    common.render_markdown("\n".join(report))


def render_metrics_and_table(
    title: str,
    metrics: Dict[str, float],
    table: pd.DataFrame,
) -> None:
    """
    Render a markdown report with metrics and a table.

    @param title: The title of the report.
    @param metrics: A dictionary of metrics to display.
    @param table: A DataFrame to display as a table.
    @return: None.
    """
    report = [
        f"#### {title}\n",
        f"- **Unique Categories:** {metrics['unique']}\n",
        f"- **Gini Coefficient:** {metrics['gini_coef']}\n",
        f"- **Top Categories (abs/rel):** {metrics['abs_top_cat']} / {metrics['rel_top_cat']}\n",
        "\n",
        table.to_markdown(index=False),
        "\n",
    ]
    common.render_markdown("\n".join(report))


def create_categorical_table(
    category_counts: pd.Series,
    top_n: Optional[int] = None,
    bottom_n: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Generate a table for a categorical column.

    @param category_counts: Value counts for the column.
    @param top_n: Number of top categories to include in the table.
    @param bottom_n: Number of bottom categories to include in the table.
    @return: A DataFrame containing the table and a dictionary of computed metrics.
    """
    # Define column names
    column_names = {"category": "Category", "count": "Count", "ratio": "Ratio"}

    # Create the full table
    full_table = pd.DataFrame(
        {
            column_names["category"]: category_counts.index,
            column_names["count"]: category_counts.values,
            column_names["ratio"]: (
                category_counts.values / sum(category_counts.values)
            ).round(4),
        }
    )

    # Slice the table if top_n and bottom_n are provided and applicable
    if top_n and bottom_n and len(category_counts) > (top_n + bottom_n + 1):
        top_table = full_table.head(top_n)
        bottom_table = full_table.tail(bottom_n)

        # Add separator row
        ellipsis_row = pd.DataFrame(
            [
                {
                    column_names["category"]: "...",
                    column_names["count"]: "...",
                    column_names["ratio"]: "...",
                }
            ]
        )
        combined_table = pd.concat(
            [top_table, ellipsis_row, bottom_table], ignore_index=True
        )
    else:
        combined_table = full_table

    return combined_table


def calculate_categorical_metrics(category_counts: pd.Series) -> Dict[str, float]:
    """
    Compute metrics for a categorical column.

    @param category_counts: Value counts for the column.
    @return: A dictionary containing the computed metrics.
    """
    # Compute proportions
    proportions = category_counts / sum(category_counts)
    gini_coefficient = 1 - sum(proportions**2)

    # Compute cumulative sum and categories for 80% of cases
    cumulative_ratios = proportions.cumsum()
    top_80_percent_count = (cumulative_ratios <= 0.8).sum() + 1
    total_categories = len(category_counts)
    relative_80_percent = (
        top_80_percent_count / total_categories if total_categories > 0 else 0
    )

    return {
        "unique": total_categories,
        "gini_coef": round(gini_coefficient, 4),
        "abs_top_cat": top_80_percent_count,
        "rel_top_cat": round(relative_80_percent, 4),
    }


def analyze_categorical_column(
    df: pd.DataFrame,
    column: str,
) -> None:
    """
    Analyze a single categorical column and append results to the report.

    @param df: The DataFrame to analyze.
    @param column: The name of the column to analyze.
    @return: None.
    """
    logging.info(f"Processing categorical column: {column}")
    category_counts = df[column].value_counts()
    table = create_categorical_table(category_counts, top_n=5, bottom_n=5)
    metrics = calculate_categorical_metrics(category_counts)

    render_metrics_and_table(
        title=f"Column: {column}",
        metrics=metrics,
        table=table,
    )

    plotting.plot_count_series(
        data=category_counts,
        title=f"{column} Distribution",
        xlabel=column,
    )


def analyze_column_dependencies(
    df: pd.DataFrame,
    column_dependencies: List[Tuple[str, str]],
) -> None:
    """
    Analyze dependencies (combined and stratified) between columns.

    @param df: The DataFrame to analyze.
    @param column_dependencies: List of (main_column, dependent_column) pairs.
    @return: None.
    """
    for main_col, dep_col in column_dependencies:
        if main_col in df.columns and dep_col in df.columns:
            logging.info(f"Processing dependency: ({main_col}, {dep_col})")

            # Combined analysis
            combined_column = df[[main_col, dep_col]].apply(tuple, axis=1)
            combined_category_counts = combined_column.value_counts()
            table = create_categorical_table(
                combined_category_counts, top_n=5, bottom_n=5
            )
            metrics = calculate_categorical_metrics(combined_category_counts)

            render_metrics_and_table(
                title=f"Combined Analysis: ({main_col}, {dep_col})",
                metrics=metrics,
                table=table,
            )

            plotting.plot_count_series(
                data=combined_category_counts,
                title=f"({main_col}, {dep_col}) Distribution",
                xlabel=f"{main_col} + {dep_col}",
                ylabel="Count",
            )

            # Stratified analysis
            common.render_markdown(
                f"#### Stratified Analysis: {dep_col} by {main_col}\n"
            )
            for main_value in df[main_col].unique():
                subset = df[df[main_col] == main_value]
                category_counts = subset[dep_col].value_counts()
                table = create_categorical_table(category_counts, top_n=5, bottom_n=5)
                metrics = calculate_categorical_metrics(category_counts)

                render_metrics_and_table(
                    title=f"{main_col} = {main_value}",
                    metrics=metrics,
                    table=table,
                )

                plotting.plot_count_series(
                    data=category_counts,
                    title=f"{dep_col} Distribution for {main_col} = {main_value}",
                    xlabel=dep_col,
                )


def generate_df_analysis_header(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
    dependencies: Optional[List[Tuple[str, str]]] = None,
) -> None:
    """
    Generate a header for the DataFrame analysis report.

    @param df: The DataFrame to analyze.
    @param exclude_columns: Columns to exclude from the summary.
    @param dependencies: List of (main_column, dependent_column) pairs to analyze.
    @return: None.
    """

    # Determine column details
    column_summaries = []
    for col in df.columns:
        column_info = {
            "Column Name": col,
            "Type": df[col].dtype.name,
            "Excluded": col in exclude_columns,
            "Dependent": any(dep[1] == col for dep in dependencies),
        }
        column_summaries.append(column_info)

    # Generate table using the tabulate library
    headers = ["Column Name", "Type", "Excluded", "Dependent"]
    rows = [
        [
            summary["Column Name"],
            summary["Type"],
            summary["Excluded"],
            summary["Dependent"],
        ]
        for summary in column_summaries
    ]
    markdown_column_report = tabulate(rows, headers=headers, tablefmt="pipe")

    # Report header
    report = [
        "### Dataset Summary\n",
        f"- Total Rows: {len(df)}\n",
        f"- Total Columns: {len(df.columns)}\n\n",
        markdown_column_report,
    ]
    common.render_markdown(report)


def generate_column_analysis(
    df: pd.DataFrame, dependencies: Optional[List[Tuple[str, str]]] = None
) -> None:
    """
    Generate analysis for each column in the DataFrame.

    @param df: The DataFrame to analyze.
    @param dependencies: List of (main_column, dependent_column) pairs to analyze.
    @return: None.
    """

    # Numerical summary
    display_numerical_summary(df)

    # Categorical summary
    cat_vars = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_vars:
        if col not in {dep[1] for dep in dependencies}:
            analyze_categorical_column(df, col)

    # Dependency analysis
    analyze_column_dependencies(df, dependencies)


def generate_df_analysis(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
    dependencies: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[str, List[dict]]:
    """
    Summarize the DataFrame, excluding specified columns and analyzing dependent columns.

    @param df: The DataFrame to summarize.
    @param exclude_columns: Columns to exclude from the summary.
    @param dependencies: List of (main_column, dependent_column) pairs to analyze.
    @return: A tuple containing the markdown report and a list of dictionaries with the column summaries.
    """
    exclude_columns = exclude_columns or []
    dependencies = dependencies or []

    # Render header
    generate_df_analysis_header(df, exclude_columns, dependencies)

    # Drop the columns to exclude (after header generation) and analyze the remaining columns
    df = df.drop(columns=exclude_columns, errors="ignore")
    generate_column_analysis(df, dependencies)


def compute_contacts_stats(
    df, stratify_by=None, filter_column=None, filter_values=None
) -> Dict:
    """
    Compute statistics for 'count_errands' in a pandas DataFrame.

    @param df: The DataFrame to analyze.
    @param stratify_by: The column to stratify the analysis by.
    @param filter_column: The column to filter the DataFrame by.
    @param filter_values: The values to filter the DataFrame by.
    @return: A dictionary containing the computed statistics.
    """
    # Apply optional filtering
    if filter_column and filter_values:
        df = df[df[filter_column].isin(filter_values)]

    # Define a function to compute descriptive statistics
    def descriptive_stats(series):
        return {
            "series": series,
            "count": series.count(),
            "mean": series.mean(),
            "std_dev": series.std(),
            "min": series.min(),
            "25th_percentile": series.quantile(0.25),
            "median": series.median(),
            "75th_percentile": series.quantile(0.75),
            "max": series.max(),
        }

    # Compute overall statistics if no stratification
    if not stratify_by:
        stats = descriptive_stats(df["count_errands"])
        return {"overall": stats}

    # Stratify data and compute statistics per group
    stratified_stats = {}
    for group_value, group_df in df.groupby(stratify_by):
        stratified_stats[group_value] = descriptive_stats(group_df["count_errands"])

    return stratified_stats
