import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def build_model(X_train, y_train):
    """
    Build and train a Linear Regression model.
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict_prices(model, X_test):
    """
    Use the trained model to predict prices.
    :param model: Trained Linear Regression model
    :param X_test: Testing features
    :return: Predicted prices
    """
    return model.predict(X_test)


def plot_regression(actual, predicted, dates, title='Linear Regression: Actual vs Predicted'):
    """
    Plot the actual and predicted close prices against dates.
    :param actual: Actual close prices (test set)
    :param predicted: Predicted close prices
    :param dates: Corresponding dates for the test set
    :param title: Title of the plot
    """
    plt.figure(figsize=(12, 6))

    # Plot actual values
    plt.plot(dates, actual, label='Actual Close', color='green')

    # Plot predicted values
    plt.plot(dates, predicted, label='Predicted Close', color='blue', linestyle='--')

    # Add labels, legend, and title
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()
