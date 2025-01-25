# Functions for building LSTM models.

import datetime as dt
import extraction as ex
import model_tester as test
import os
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Creates sequences for model creation.
def create_model_sequences(csv_path, seq_length):

    # Creating dictionary with training and test information about articles.
    article_info = ex.extract_all_articles_info_hot_encoded(csv_path, \
        'data/json/article_dictionary.json')
    
    # Creating numpy array with sequences.
    sequences = ex.create_dictionary_sequences(article_info, seq_length)

    # Returning sequences.
    return sequences

# Creating lstm neural network based on input data.
def build_lstm_neural_network(sequences, seq_length, epochs_no, loss_func, batch, neurons, \
    learning, features, dropout):        
    
    # Defining model.
    model = Sequential()

    # Single layer network.
    if len(neurons) == 1:

        # Input layer.
        model.add(LSTM(neurons[0], activation='tanh', input_shape=(seq_length, features)))
        model.add(Dropout(dropout))       
    
    # Multiple layer network.
    if len(neurons) > 1:

        # Input layer.
        model.add(LSTM(neurons[0], activation='tanh', input_shape=(seq_length, features), return_sequences=True))
        model.add(Dropout(dropout))   

        # Hidden layers except last.
        if len(neurons) > 2:
            for i in (1, len(neurons)-1):
                model.add(LSTM(neurons[i], activation='tanh', return_sequences=True))
                model.add(Dropout(dropout))

        # Last hidden layer.
        model.add(LSTM(neurons[-1], activation='tanh'))
        model.add(Dropout(dropout))
        
    # Output layer     
    model.add(Dense(1))
    
    # Compiling model.
    model.compile(optimizer=Adam(learning_rate=learning), loss=loss_func)

    # Printing model summary.
    model.summary()

    # Recording and printing start time.
    start_time = dt.datetime.today()
    print(f"Start time: {start_time.strftime('%H:%M:%S')}")

    # Training model.
    model.fit(sequences['training_input'], sequences['training_output'], epochs=epochs_no, \
        batch_size=batch)
    
    # Recording and printing completion time.
    completion_time = dt.datetime.today()
    print(f"Completion time: {completion_time.strftime('%H:%M:%S')}")
    
    # Printing total training time.
    print(f"Total training time: {completion_time-start_time}")
    
    # Returning complete model.
    return model


# Saving model with descriptive title.
def save_model(model, seq_length, epochs_no, loss_func, batch, neurons, \
    learning, features, dropout, model_path= ""):
    
    # Creating string describing the number of neurons in model.
    # Creating string describing number of neurons in model.
    neuron_string = ex.list_item_string(neurons)

    # Defining model name.
    # Format: sequence length-number of epochs-loss function-batch size-number of neurons in each layer.
    model_name = f"{seq_length}seq-{epochs_no}epochs-{loss_func}-{batch}batch-{neuron_string}neurons-{learning}lr-{features}features-{dropout}dr".replace(".","_")
    
    # Creating folder to store model in.
    os.makedirs(f"models/{model_path}{model_name}")

    # Saving model.
    model.save(f"models/{model_path}{model_name}/{model_name}.keras")

    # Returning model name.
    return model_name


# Creating, saving and testing LSTM neural network.
# Use create_model_sequences(csv_path, seq_length) to create sequences.
# The number of neurons is entered as items in a list: e.g., [176] or [176, 88].
# Model path specifices subdirectory of models/ and is entered as string: e.g., "model_path/".
def create_lstm_neural_network(sequences, seq_length, epochs_no, loss_func, batch, neurons, \
        learning, features, dropout, model_path=""):   

    # Creating model.
    model = build_lstm_neural_network(sequences, seq_length, epochs_no, loss_func, batch, neurons, \
        learning, features, dropout)
    
    # Saving model and retrieving model name.
    model_name = save_model(model, seq_length, epochs_no, loss_func, batch, neurons, learning, \
        features, dropout, model_path)

    # Testing model.
    test.test_model(model, sequences, model_name, seq_length, epochs_no, loss_func, batch, \
        neurons, learning, features, dropout, model_path)
   