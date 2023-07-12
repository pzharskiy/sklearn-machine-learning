import matplotlib.pyplot as plt
import seaborn as sns


def linechart(data, figsize):
    # Set the width and height of the figure
    plt.figure(figsize=figsize)

    sns.lineplot(data=data)
    plt.show()


def barchart(data, x, y, title, rotation="vertical"):
    sns.barplot(data=data, x=x, y=y)
    # How labels for x-axis will be presented:
    plt.xticks(rotation=rotation)
    plt.title(title)
    plt.show()


def heatmap(data, figsize, horizontal_label, title, annotations=True):
    plt.figure(figsize=figsize)
    sns.heatmap(data, annot=annotations)
    # Add label for horizontal axis
    plt.xlabel(horizontal_label)
    # Add label for vertical axis
    plt.title(title)
    plt.show()


def scatter_plot(x, y):
    sns.scatterplot(x=x, y=y)
    plt.show()


def regression_scatter_plot(x, y):
    sns.regplot(x=x, y=y)
    plt.show()


def scatter_plot_colored(x, y, hue):
    sns.scatterplot(x=x, y=y, hue=hue)
    plt.show()


def scatter_plot_colored_with_multiple_regression_lines(x, y, hue, data):
    sns.lmplot(x=x, y=y, hue=hue, data=data)
    plt.show()


def categorical_scatter_plot(x, y):
    sns.swarmplot(x=x, y=y)
    plt.show()
    #60.8% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.


def categorical_strip_plot(x, y):
    sns.stripplot(x=x, y=y)
    plt.show()
    #Just like alternative to sw, due to console warning


def histogram(data_column):
    sns.histplot(data=data_column)
    plt.show()


def colored_histogram(data, x, hue, title):
    sns.histplot(data=data, x=x, hue=hue)
    plt.title(title)
    plt.show()


def kernel_density_estimate(data_column, shade=True):
    sns.kdeplot(data=data_column, shade=shade)
    plt.show()


def colored_kernel_density_estimate(data, x, hue, title, fill=True):
    # KDE plots for each species
    sns.kdeplot(data=data, x=x, hue=hue, fill=True)
    plt.title(title)
    plt.show()


def two_dimensional_kernel_density_estimate(x,y, kind="kde"):
    sns.jointplot(x=x, y=y, kind=kind)
    plt.show()