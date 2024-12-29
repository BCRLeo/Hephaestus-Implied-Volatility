import numpy as np
import math

def scale_range(x, input_range, target_range):
    """
    Rescale a numpy array from input to target range
    :param x: data to scale
    :param input_range: [min, max] of original data
    :param target_range: [min, max] of new scale
    :return: x_scaled, original_range
    """
    range_ = [np.amin(x), np.amax(x)]
    x_std = (x - input_range[0]) / float(input_range[1] - input_range[0])
    x_scaled = x_std * float(target_range[1] - target_range[0]) + target_range[0]
    return x_scaled, range_


def train_test_split_linear_regression(stocks):
    """
    Split the data set into training and testing sets for a Linear Regression Model.
    
    We will use 'Open' as our single feature and 'Close' as our label (target).
    You can modify these columns as desired.

    :param stocks: DataFrame containing columns like ['Open','Close', ...].
    :return: X_train, X_test, y_train, y_test, label_range
    """

    # Prepare lists for features and labels
    feature = []
    label = []

    # Convert relevant columns to numpy arrays
    # In your VIX DataFrame, 'Open' is the feature, 'Close' is the label
    for index, row in stocks.iterrows():
        feature.append([row['Open']])   # single feature
        label.append([row['Close']])    # single label

    # Determine min/max for feature and label
    feature_min = min(f[0] for f in feature)
    feature_max = max(f[0] for f in feature)
    label_min   = min(l[0] for l in label)
    label_max   = max(l[0] for l in label)

    # Scale features and labels to [-1.0, 1.0]
    feature_scaled, feature_range = scale_range(
        np.array(feature),
        input_range=[feature_min, feature_max],
        target_range=[-1.0, 1.0]
    )
    label_scaled, label_range = scale_range(
        np.array(label),
        input_range=[label_min, label_max],
        target_range=[-1.0, 1.0]
    )

    # Decide how much data to use for testing (here 31.5% as in the original code)
    split_ratio = 0.315
    split = int(math.floor(len(feature_scaled) * split_ratio))

    # Split into train and test sets
    X_train = feature_scaled[:-split]
    X_test  = feature_scaled[-split:]

    y_train = label_scaled[:-split]
    y_test  = label_scaled[-split:]

    return X_train, X_test, y_train, y_test, label_range


def train_test_split_lstm(stocks, prediction_time=1, test_data_size=450, unroll_length=50):
    """
    (unchanged, unless you want to adapt columns)
    """
    test_data_cut = test_data_size + unroll_length + 1

    # NOTE: as_matrix() is deprecated; prefer .values or .to_numpy() in modern pandas
    x_train = stocks[0:-prediction_time - test_data_cut].to_numpy()
    y_train = stocks[prediction_time:-test_data_cut]['Close'].to_numpy()

    x_test = stocks[-test_data_cut:-prediction_time].to_numpy()
    y_test = stocks[prediction_time - test_data_cut:]['Close'].to_numpy()

    return x_train, x_test, y_train, y_test

def unroll(data, sequence_length=24):
    """
    (unchanged)
    """
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)
