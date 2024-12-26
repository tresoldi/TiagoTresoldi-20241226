import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


def plot_count_series(
    data: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str = "Count",
    kind: str = "bar",
    xticks_rotation: int = 45,
    max_categories: int = 20,
    label_max_length: int = 16,
    show_percentage: bool = False,
    show_decimals: bool = False,
    show_other: bool = True,
    confidence_level: float = None,
) -> None:
    """
    Prepare and display a plot with enhancements for labels, percentages, confidence intervals, and aesthetics in a Jupyter Notebook.

    @param data: The data to be plotted as a Pandas Series.
    @param title: The title of the plot.
    @param xlabel: The label for the x-axis.
    @param ylabel: The label for the y-axis (default: "Count").
    @param kind: The type of plot (default: "bar").
    @param xticks_rotation: The rotation angle for the x-axis labels (default: 45).
    @param max_categories: The maximum number of categories to display in the plot (default: 20).
    @param label_max_length: The maximum length for x-axis labels before truncation (default: 14).
    @param show_percentage: Whether to show numbers as percentages (default: False).
    @param show_decimals: Whether to show numbers with two decimal places (default: False).
    @param show_other: Whether to combine categories beyond max_categories into an "Other" category (default: True).
    @param confidence_level: Confidence level for confidence intervals (e.g., 0.95). If None, confidence intervals are not shown.
    @return: None
    """

    # Prepare data
    if len(data) > max_categories:
        top_data = data.head(max_categories - 1)
        other_count = data.iloc[max_categories - 1 :].sum()
        if show_other:
            data = pd.concat([top_data, pd.Series({"Other": other_count})])
        else:
            data = top_data

    if show_percentage:
        data = (data / data.sum()) * 100

    truncated_labels = [
        label[:label_max_length] for label in data.index
    ]
    color_list = [plt.cm.tab20.colors[i % 20] for i in range(len(data))]

    # Confidence intervals
    if confidence_level and not show_percentage:
        total = data.sum()
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        errors = [
            z * ((x / total) * (1 - (x / total)) / total) ** 0.5 if total > 0 else 0
            for x in data
        ]
    else:
        errors = None

    # Render plot
    plt.figure(figsize=(max(12, len(data) * 0.5), 6))
    if kind == "bar" and errors:
        plt.bar(data.index, data, color=color_list, yerr=errors, capsize=5)
    else:
        data.plot(kind=kind, title=f"{title} (Total: {data.sum():.0f})", color=color_list)

    plt.xlabel(xlabel, fontsize=12, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12, fontweight="bold")
    plt.xticks(
        ticks=range(len(truncated_labels)),
        labels=truncated_labels,
        rotation=xticks_rotation,
    )
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add data labels
    for i, value in enumerate(data):
        plt.text(
            i,
            value,
            f"{value:.2f}" if any([show_percentage, show_decimals]) else f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Show plot
    plt.tight_layout()
    plt.show()



def plot_stratified_bar(stats, metric="mean", title="Stratified Bar Plot"):
    """
    Plot a bar chart for a specified metric from the stratified statistics.

    Parameters:
        stats (dict): The output of compute_contacts_stats when stratified.
        metric (str): The metric to plot (e.g., 'mean', 'median').
        title (str): The title of the plot.
    """
    groups = list(stats.keys())
    values = [stats[group][metric] for group in groups]

    plt.figure(figsize=(10, 6))
    plt.bar(groups, values, alpha=0.7)
    plt.xlabel("Group")
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_stratified_proportions(stats, title="Proportions of Errand Counts"):
    """
    Plot a grouped bar chart for the proportions of errand counts (binned into categories).

    Parameters:
        stats (dict): The output of compute_contacts_stats when stratified.
        title (str): The title of the plot.
    """
    # Define bins and labels
    bins = [0, 1, 2, 3, 4, float("inf")]
    labels = ["0", "1", "2", "3", "4"]

    # Prepare data for plotting
    proportions = {}
    for group, data in stats.items():
        # Bin the series and compute proportions
        binned = pd.cut(data["series"], bins=bins, labels=labels, right=False)
        proportions[group] = binned.value_counts(normalize=True).sort_index()

    # Convert to DataFrame for easy plotting
    proportions_df = pd.DataFrame(proportions).T  # Transpose for better formatting

    # Plot grouped bar chart
    proportions_df.plot(kind="bar", stacked=False, figsize=(12, 6), alpha=0.8)
    plt.xlabel("Groups")
    plt.ylabel("Proportion")
    plt.title(title)
    plt.legend(title="Errand Count Categories")
    plt.tight_layout()
    plt.show()
