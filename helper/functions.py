import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_histogram_outlier(dataframe, feature):
    '''
    This function is used to plot a histogram and a boxplot for a specified feature in a pandas DataFrame. 
    The function is intended to help visualize the distribution of the data and to detect any outliers.

    Parameters:
    -----------
    dataframe : pandas DataFrame
        The DataFrame containing the data to be plotted.
    feature : str
        The name of the feature to be plotted.

    Returns:
    --------
    None

    Examples:
    ---------
    >>> import seaborn as sns
    >>> df = sns.load_dataset('iris')
    >>> plot_histogram_outlier(df, 'sepal_length')
    '''

    # Create a figure with two subplots: a histogram and a boxplot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Plot a histogram of the specified feature using seaborn's histplot function
    sns.histplot(data=dataframe, x=feature, ax=ax[0], kde=True)
    sns.set_style('darkgrid')
    sns.set_palette('Accent')
    ax[0].set_title(f'{feature.title()} Histogram')

    # Plot a boxplot of the specified feature using seaborn's boxplot function
    sns.boxplot(data=dataframe, x=feature, ax=ax[1])
    sns.set_style('darkgrid')
    sns.set_palette('vlag')
    ax[1].set_title(f'{feature.title()} Boxplot')

    # Show the plots
    plt.show()

def plot_correlation_heatmap(correlation_matrix, cmap_color):
    '''
    This function is used to plot given correlation matrix into heatmap to better see the features correlation

    Parameters:
    -----------
    dataframe: pandas Dataframe
        The DataFrame containing the data to be plotted.

    Returns:
    --------
    None

    Examples:
    ---------
    >>> import seaborn as sns
    >>> df = sns.load_dataset('iris')
    >>> correlation_matrix = df[numerical_features].corr()
    >>> plot_correlation_heatmap(correlation_matrix)
    '''
    plt.figure(figsize=(7, 5))

    sns.heatmap(data=correlation_matrix, annot=True, cmap=cmap_color, linewidths=0.5)
    plt.title('Correlation Matrix for Numerical Features')
    plt.show()
