import matplotlib.pyplot as plt

# Set default figure size
plt.rcParams['figure.figsize'] = (18, 12)


def price(x):
    """
    Format the coords message box
    :param x: data to be formatted
    :return: formatted data
    """
    return f'${x:1.2f}'


def plot_basic(stocks, title='VIX Trading', y_label='VIX Level', x_label='Date'):
    """
    Plots basic pyplot for VIX data
    :param stocks: DataFrame having all the necessary data
    :param title: Title of the plot 
    :param y_label: Y-label of the plot
    :param x_label: X-label of the plot
    :return: Displays a Pyplot against the time index and their closing value
    """
    fig, ax = plt.subplots()
    
    # Ensure the index is used for x-axis
    ax.plot(stocks.index, stocks['Close'], color='#0A7388', label='Close')

    ax.format_ydata = price
    ax.set_title(title)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()

    plt.show()


def plot_prediction(dates, actual, prediction, title='VIX Actual vs Prediction', y_label='Close Price', x_label='Date'):
    """
    Plots actual vs predicted values with dates on the x-axis.
    :param dates: Array-like containing dates corresponding to the predictions
    :param actual: Array-like containing actual data (e.g., Close prices)
    :param prediction: Array-like containing predicted values
    :param title: Title of the plot
    :param y_label: Y-label of the plot
    :param x_label: X-label of the plot
    :return: Displays a Pyplot with actual and predicted closing values
    """
    plt.figure(figsize=(12, 6))

    # Plot actual values
    plt.plot(dates, actual, label='Actual Close', color='green')

    # Plot predicted values
    plt.plot(dates, prediction, label='Predicted Close', color='blue', linestyle='--')

    # Add labels, legend, and title
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_lstm_prediction(actual, prediction, title='VIX LSTM Prediction', y_label='VIX Level', x_label='Date'):
    """
    Plots LSTM train, test, and prediction
    :param actual: Array-like containing actual data
    :param prediction: Array-like containing predicted values
    :param title: Title of the plot
    :param y_label: Y-label of the plot
    :param x_label: X-label of the plot
    :return: Displays a Pyplot with actual and predicted closing values
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # Plot actual and predicted values
    plt.plot(actual, color='#00FF00', label='Actual Close')
    plt.plot(prediction, color='#FF5733', linestyle='--', label='LSTM Predicted Close')

    # Set title
    ax.set_title(title)
    ax.legend(loc='upper left')

    plt.show()
