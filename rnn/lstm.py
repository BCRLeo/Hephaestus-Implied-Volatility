from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential


def build_improved_model(input_dim, output_dim, return_sequences=True):
    """
    Builds an improved Long Short Term Memory (LSTM) model.
    :param input_dim: Input dimension of the model (number of features).
    :param output_dim: Output dimension of the model (number of LSTM units).
    :param return_sequences: Whether the first LSTM layer should return sequences.
    :return: A 3-layered LSTM model.
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(
        units=output_dim,
        input_shape=(None, input_dim),
        return_sequences=return_sequences))
    model.add(Dropout(0.2))

    # Second LSTM layer
    model.add(LSTM(
        units=128,
        return_sequences=False))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    return model


def build_basic_model(input_dim, output_dim, return_sequences=True):
    """
    Builds a basic LSTM model.
    :param input_dim: Input dimension of the model (number of features).
    :param output_dim: Output dimension of the model (number of LSTM units).
    :param return_sequences: Whether the first LSTM layer should return sequences.
    :return: A basic LSTM model.
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(
        units=output_dim,
        input_shape=(None, input_dim),
        return_sequences=return_sequences))

    # Second LSTM layer
    model.add(LSTM(
        units=100,
        return_sequences=False))

    # Output layer
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    return model
