import neural_network_builder as nnb

if __name__ == "__main__":
    
    # Create sequences.
    sequences = nnb.create_model_sequences('training_data/325_articles.csv', 5)
    
    # Create, save and test LSTM network.
    # Model path for saving is derived from specified hyperparameters.
    # Subdirectory of models folder can be specified in model_path parameter.
    # Plots and prints are stored in model directory.
    nnb.create_lstm_neural_network(sequences, seq_length=5, epochs_no=5, loss_func='mse', \
    batch=32, neurons=[351], learning=0.01, features=351, dropout=0.1, model_path="testing/")