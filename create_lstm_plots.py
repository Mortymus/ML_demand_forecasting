import model_tester as test

if __name__ == "__main__":

    # Create list with interesting articles to plot.
    interesting_articles = ["704285301", "704285801", "704197401"]

    # Create predictions and plot results.
    # Model path is derived from specified hyperparameters.
    # Subdirectory of models folder can be specified in model_path parameter.
    # Plots will be stored in model directory.
    test.plot_articles_from_saved_model(article_nos=interesting_articles, \
        seq_length=5, epochs_no=5, loss_func='mse', batch=32, neurons=[176], \
        learning=0.01, features=351, dropout=0.1, model_path="testing/")