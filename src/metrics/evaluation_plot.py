import colorcet as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configuration settings for plots
SCATTER_SIZE = 25
COLOR_PALETTE = sns.color_palette(cc.glasbey)

NEIGHBORHOOD_ENRICHMENT_CMAP = "inferno"
NEIGHBORHOOD_ENRICHMENT_FIGSIZE = (5, 5)
DPI = 300


def set_plot_legend(
    ax: plt.Axes,
    ncol: int = 6,
    location: str = "upper center",
    bbox_anchor=(0.5, -0.05),
) -> None:
    """
    Utility function to set the legend for a plot.

    Parameters:
    ax (plt.Axes): The Axes object of the plot.
    ncol (int, optional): Number of columns in the legend. Defaults to 6.
    location (str, optional): The location of the legend. Defaults to "upper center".
    bbox_anchor (tuple, optional): The bbox anchor for the legend. Defaults to (0.5, -0.05).
    """

    ax.legend(
        loc=location,
        bbox_to_anchor=bbox_anchor,
        fancybox=True,
        shadow=True,
        ncol=ncol,
    )


def save_plot(fig: plt.Figure, path: str, dpi: int = DPI) -> None:
    """
    Utility function to save a plot.

    Parameters:
    fig (plt.Figure): The Figure object to save.
    path (str): The file path to save the figure.
    dpi (int, optional): The resolution of the figure in dots per inch. Defaults to DPI.
    """

    plt.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def create_scatter_plot(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    hue_order: list,
) -> None:
    """
    Creates a scatter plot on the given Axes.
    Parameters:
    ax (plt.Axes): The Axes object to create the plot on.
    data (pd.DataFrame): The DataFrame containing the data to plot.
    x_col (str): The name of the column to use for x-axis values.
    y_col (str): The name of the column to use for y-axis values.
    hue_col (str): The name of the column to define the color of the points.
    hue_order (list): The order of the hues.
    """

    sns.scatterplot(
        data,
        x=x_col,
        y=y_col,
        hue=hue_col,
        s=SCATTER_SIZE,
        ax=ax,
        hue_order=hue_order,
        palette=sns.color_palette(COLOR_PALETTE),
    )


def plot_scatter_visualization(metadata_true, metadata_pred, uniques, dir):
    fig, axarr = plt.subplots(1, 2, figsize=(16, 6))

    # Define the color palette for the unique categories
    pl_palette = sns.color_palette(cc.glasbey, n_colors=len(uniques))
    palette_dict = dict(zip(uniques, pl_palette))

    # Groundtruth Plot
    axarr[0].set_title("Groundtruth", fontsize=16)  # Adjust title font size
    g1 = sns.scatterplot(
        data=metadata_true,
        x="coord_X",
        y="coord_Y",
        hue="cell_class",
        s=15,
        ax=axarr[0],
        palette=palette_dict,
        legend=False,  # Do not automatically create a legend
    )
    g1.set_xlabel("X", fontsize=14)  # Adjust X-axis label font size
    g1.set_ylabel("Y", fontsize=14)  # Adjust Y-axis label font size

    # Prediction Plot
    axarr[1].set_title("Prediction", fontsize=16)  # Adjust title font size
    g2 = sns.scatterplot(
        data=metadata_pred,
        x="coord_X",
        y="coord_Y",
        hue="cell_class",
        s=15,
        ax=axarr[1],
        palette=palette_dict,
        legend=False,  # Do not automatically create a legend
    )
    g2.set_xlabel("X", fontsize=14)  # Adjust X-axis label font size
    g2.set_ylabel("Y", fontsize=14)  # Adjust Y-axis label font size

    # Create custom legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=cat,
            markerfacecolor=palette_dict[cat],
            markersize=10,
        )
        for cat in uniques
    ]
    ncol = len(uniques) // 4  # Number of columns in the legend
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.5, -0.05),
    )

    # Save the plot
    path = f"{dir}/class_scatter_plot.pdf"
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
