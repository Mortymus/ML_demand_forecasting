import model_tester as test

if __name__ == "__main__":

    # Running tests on saved model.
    # Creating predictions to plot and print statistics.
    # Model path is derived from specified hyperparameters.
    # Subdirectory of models folder can be specified in model_path parameter.
    # Plots and prints are stored in model directory.
    test.test_saved_model(seq_length=5, epochs_no=5, loss_func='mse', batch=32, \
        neurons=[176], learning=0.01, features=351, dropout=0.1, model_path="testing/")