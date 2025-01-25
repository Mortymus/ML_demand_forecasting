# Runs and tests models to create statistics with plots and prints.

import datetime as dt
import extraction as ex
from itertools import chain
import keras
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from operator import itemgetter
import os
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

#########################
### Support functions ###
#########################

# Translate loss function.
def loss_function_translation(loss_func):   

    # Creating loss function dictionary.
    loss_functions = {
        'mae': "mean absolute error",
        'mse': "mean squared error",
        'mape': "mean absolute percentage error",
        'Huber()': "huber"
        }
    
    # Returning applicable function if available.
    try:
        return loss_functions[loss_func]
    
    # Returning unknown if not available.
    except:
        return "unknown function"
    

# Creates string with numbers of neurons from list.
def list_neurons(neurons):

    # Create string for neurons with 1st instance of list.
    neuron_string = f"{neurons[0]}"

    # Adding remaining instances if any.
    if len(neurons) > 1:
        for i in range(1, len(neurons)):
            neuron_string += f", {neurons[i]}"
    
    # Returning created string.
    return neuron_string


# Creates string with model name.
def create_model_name(seq_length, epochs_no, loss_func, batch, neurons, \
    learning, features, dropout):

    # Creating string with list of neurons.
    # Creating string describing number of neurons in model.
    neuron_string = ex.list_item_string(neurons)

    # Returning string with model name.
    return f"{seq_length}seq-{epochs_no}epochs-{loss_func}-{batch}batch-{neuron_string}neurons-{learning}lr-{features}features-{dropout}dr".replace(".","_")


# Grouping predictions by articles.
def group_predictions_by_article(sequences_input, sequences_output, predictions):

    # Creating dictionary for storing predictions with article number as key.
    predictions_by_article = {}

    # Loading reverse lookup article dictionary.
    article_dict_lookup = ex.load_json_dictionary('data/json/article_dictionary_lookup.json')

    # Creating variable storing article number and current index.
    # Starting on first article index with corresponding article number and initialized dictionary.
    current_index = 0
    article_number = "704197901"
    predictions_by_article[article_number] = {'actual_sale': [], 'prediction': []}

    for i in range(len(sequences_input)):

        # Checking if entry has same article number as last.
        if sequences_input[i][0][current_index] != 1:
        
            # Checking one hot encoded article number if article has changed.
            for j in range(0, 325):
                if sequences_input[i][0][j] == 1:

                    # Translating one-hot encoded article number to article number translation.
                    article_number_translation = j + 1

                    # Fetching original article number.
                    article_number = article_dict_lookup[str(article_number_translation)]['article_number']

                    # Updating current index to one-hot encoded article index.
                    current_index = j

                    # Creating dictionary key with lists if it does not exist.
                    if article_number not in predictions_by_article:
                        predictions_by_article[article_number] = {'actual_sale': [], 'prediction': []}
                    
                    # Discontinuing loop since article number is found.
                    break
        
        predictions_by_article[article_number]['actual_sale'].append(sequences_output[i])    
        predictions_by_article[article_number]['prediction'].append(predictions[i][0])
    
    # Returning dictionary with predictions grouped by article.
    return predictions_by_article


# Checking performance of predictions for all articles.
def check_total_performane(sequences_output, predictions):

    # Creating dictionary with total performance of model.
    total_performance = {
        'total_mape': mean_absolute_percentage_error(sequences_output, predictions),
        'total_mse': mean_squared_error(sequences_output, predictions),
        'total_mae': mean_absolute_error(sequences_output, predictions),
        'total_r2': r2_score(sequences_output, predictions)
    }

    # Returning dictionary with total performance.
    return total_performance


# Checks performance of individual articles and sort them by r squared values.
def check_and_sort_article_performance(predictions_by_article):
    
    # Creating list with statistics for articles.
    # [article number, mape, mse, mae, r2]
    # [0] = article number
    # [1] = mean absolute percentage error
    # [2] = mean squared error
    # [3] = mean absolute error
    # [4] = r squared
    article_stats = []

    # Checking accuracy by article and appending to list.
    for article in predictions_by_article:
        article_sales = predictions_by_article[article]['actual_sale']
        article_predictions = predictions_by_article[article]['prediction']
        mape = mean_absolute_percentage_error(article_sales, article_predictions)
        mse = mean_squared_error(article_sales, article_predictions)
        mae = mean_absolute_error(article_sales, article_predictions)
        r2 = r2_score(article_sales, article_predictions)
        article_stats.append([article, mape, mse, mae, r2])

    # Sorting article stats by highest r2 score.
    article_stats_sorted = sorted(article_stats, key=itemgetter(4), reverse=True)

    # Returning article statistics sorted by r squared score.
    return article_stats_sorted


# Create prediction statistics and plots.
# Model path specifices subdirectory of models/ and is entered as string: "model_path/".
# Neurons are entered per layer as integers in a list.
def create_prediction_statistics(model, sequences_input, sequences_output, \
        article_dict, seq_length, epochs_no, loss_func, batch, neurons, \
        learning, features, dropout, model_name, dataset, model_path):

    # Making predictions.
    predictions = model.predict(sequences_input)

    # Checking total model performance.
    total_performance = check_total_performane(sequences_output, predictions)

    # Grouping predictions by article.
    predictions_by_article = group_predictions_by_article(sequences_input, \
        sequences_output, predictions)

    # Checking accuracy for individual articles and sorting by r squared value.
    # [article number, mape, mse, mae, r2]
    # [0] = article number
    # [1] = mean absolute percentage error
    # [2] = mean squared error
    # [3] = mean absolute error
    # [4] = r squared
    article_stats_sorted = check_and_sort_article_performance(predictions_by_article)

    # Plotting total predictions compared to sales.
    plot_total_predictions(sequences_output, predictions, seq_length, epochs_no, loss_func,\
        batch, neurons, learning, features, dropout, model_name, total_performance, dataset, \
        model_path)

    # Plotting predictions compared to sales for top and bottom 5 articles.
    plot_article_predictions(article_dict, predictions_by_article, article_stats_sorted, \
        seq_length, epochs_no, loss_func, batch, neurons, learning, features, dropout, \
        model_name, dataset, model_path)
    
    # Print model statistics to txt file.
    print_model_statistics_to_txt(article_dict, model_name, seq_length, epochs_no, loss_func, \
        batch, neurons, learning, features, dropout, total_performance, article_stats_sorted, \
        dataset, model_path)
    

# Create prediction plot for specific articles.
# Model path specifices subdirectory of models/ and is entered as string: "model_path/".
# Neurons are entered per layer as integers in a list.
def create_prediction_plot_articles(model, sequences_input, sequences_output, article_nos, \
    article_dict, seq_length, epochs_no, loss_func, batch, neurons, learning, features, \
    dropout, model_name, dataset, model_path):

    # Making predictions.
    predictions = model.predict(sequences_input)

    # Grouping predictions by article.
    predictions_by_article = group_predictions_by_article(sequences_input, sequences_output, \
        predictions)
    
    # Plotting article predictions for test set.
    plot_specific_article_predictions(article_nos, article_dict, predictions_by_article, \
        seq_length, epochs_no, loss_func, batch, neurons, learning, features, dropout, \
        model_name, dataset, model_path)
    

# Loops through root directory gathering model stats.
# Enter "test" or "training" for dataset.
def rank_rootdir_models(directory, dataset, excepted_dirs={}):
 
    # Creating list to store model stats.
    model_stats = []

    # Looping through root directory and subdirectories.
    for dirpath, dirnames, filenames in os.walk(directory):

        # Excluding excepted directories from loop.
        if dirpath == directory:
            dirnames[:] = [dir for dir in dirnames if dir not in excepted_dirs]

        # Looping through files in current directory.
        for file in filenames:

            # Retrieving stats from test and training txt files.
            if file.endswith(f"{dataset}.txt"):
                with open(f"{dirpath}/{file}", 'r', encoding='utf-8') as stats:
                    lines = stats.readlines()
                    info = lines[3:18]
                    r2 = float(lines[16][19:-2])
                    model_stats.append([r2, info])

    # Sorting model performance by r2 score.
    sorted_model_stats = sorted(model_stats, key=lambda x: x[0], reverse=True)
    
    # Writing rank of models to txt file.
    with open(f"{directory}/ranked_{dataset}_models.txt", 'w', encoding='utf-8') as output:

        # Creating headline pointing to type of dataset.
        output.write(f"Ranking models by {dataset} scores\n")

        # Crating counter for ranking.
        counter = 1

        # Looping through all models.
        for model in sorted_model_stats:

            # Creating string with statistics-
            model_string = "".join(model[1])

            # Writing statistics to file.
            output.write(f"{counter}. ranked {dataset} model\n")
            output.write(model_string)

            # Updating counter.
            counter += 1


# Loops through root directory gathering model configurations.
def sort_rootdir_models(directory, excepted_dirs={}):
 
    # Creating list to store model configurations.
    model_configs = []

    # Looping through root directory and subdirectories.
    for dirpath, dirnames, filenames in os.walk(directory):

        # Excluding excepted directories from loop.
        if dirpath == directory:
            dirnames[:] = [dir for dir in dirnames if dir not in excepted_dirs]

        # Looping through files in current directory.
        for file in filenames:

            # Retrieving configurations from test txt files.
            if file.endswith(f"test.txt"):
                with open(f"{dirpath}/{file}", 'r', encoding='utf-8') as stats:
                    lines = stats.readlines()
                    seq_length = int(lines[4][25:-1])
                    epochs = int(lines[5][25:-1])
                    batch = int(lines[8][20:-1])
                    neurons = lines[9][27:-1]
                    learning_rate = float(lines[6][22:-1])
                    features = int(lines[3][28:-1])
                    dropout_rate = float(lines[10][22:-1])
            
                # Creating list to store model configurations.
                model = [
                    seq_length,
                    epochs,
                    batch,
                    neurons,
                    learning_rate,
                    features,
                    dropout_rate
                ]

                # Adding model to list of models.
                model_configs.append(model)
                    

    # Sorting model configurations.
    sorted_model_configs = sorted(model_configs, key=lambda \
        x: (x[5], x[0], x[3], x[2], x[4], x[1], x[6]))
    
    # Writing configurations to txt file.
    with open(f"{directory}/sorted_models.txt", 'w', encoding='utf-8') as output:

        # Creating headline.
        output.write("Sorted models\n\n")

        # Crating counter.
        counter = 1

        # Looping through all models.
        for model in sorted_model_configs:

            # Creating string with configurations.
            model_string = f"""
Sequence length:    {model[0]}
Number of epocs:    {model[1]}
Batch size:         {model[2]}
Neurons:            {model[3]}
Learning rate:      {model[4]}
Features:           {model[5]}
Dropout rate:       {model[6]}\n
"""

            # Writing configurations to file.
            output.write(f"{counter}. {model_string}")

            # Updating counter.
            counter += 1


##################################################
### Statistics printing and plotting functions ###
##################################################

# Creeating title string for plot.
def create_title_string(features, seq_length, epochs_no, learning, loss_func, \
    batch, neurons, dropout):

    # Returining title string.
    return f"""Number of features: {features}
Sequence length: {seq_length}
Number of epochs: {epochs_no}
Learning rate: {learning}
Loss function: {loss_function_translation(loss_func)}
Batch size: {batch}
Number of neurons: {list_neurons(neurons)}
Dropout rate: {dropout}"""

# Creating x-label for plot.
def create_xlabel_string(mae, mape, mse, r2):

    # Returning x-label string.
    return f"""Sequences
Mean absolute error: {mae:.2f}
Mean absolute percentage error: {mape:.2f}
Root mean squared error: {mse**0.5:.2f}
Coefficient of determination: {r2:.4f}"""


# Prints total model and individual article statistics to txt file.
# Model path specifices subdirectory of models/ and is entered as string: "model_path/".
# Neurons are entered per layer as integers in a list.
def print_model_statistics_to_txt(article_dict, model_name, seq_length, epochs_no, loss_func, \
    batch, neurons, learning, features, dropout, total_performance, article_stats_sorted, dataset, model_path=""):

    # Creating file with statistices for model by article.
    with open(f'models/{model_path}{model_name}/{model_name}_{dataset}.txt', 'w', encoding='utf-8') as txt_file:
        
        # Adding headline and total statistics for model.
        txt_file.write(f"""{dataset.capitalize()} dataset for all articles
                       
        LSTM hyperparameters:
        Number of features: {features}                       
        Sequence length: {seq_length}
        Number of epochs: {epochs_no}
        Learning rate: {learning}
        Loss function: {loss_func}
        Batch size: {batch}
        Number of neurons: {list_neurons(neurons)}
        Dropout rate: {dropout}
            
        Test results:
        Mean absolute percentage error: {total_performance['total_mape']}
        Root mean squared error: {total_performance['total_mse']**0.5}
        Mean absolute error: {total_performance['total_mae']}
        R squared: {total_performance['total_r2']}\n\n""")
        
        # Adding individual statistics for each article.
        counter = 1
        for article in article_stats_sorted:
            txt_file.write(f"""{counter}. Article number: {article[0]}
        Article name: {article_dict[article[0]]['name']}
        Mean absolute percentage error: {article[1]}
        Root mean squared error: {article[2]**0.5}
        Mean absolute error: {article[3]}
        R squared: {article[4]}\n""")
            counter += 1


# Plots actual sales compared to predictions for entire model.
# Model path specifices subdirectory of models/ and is entered as string: "model_path/".
# Neurons are entered per layer as integers in a list.
def plot_total_predictions(sequences_output, predictions, seq_length, epochs_no, loss_func,\
    batch, neurons, learning, features, dropout, model_name, total_performance, dataset, \
    model_path=""):

    # total_performance = {'total_mae', 'total_mape', 'total_mse', 'total_r2'}
    
    # Setting plot size.
    plt.figure(figsize=(10,8))

    # Plotting predictions vs. sales for entire model.
    plt.plot(predictions, label="Predictions", linestyle=(0, (5, 10)), zorder=2)
    plt.plot(sequences_output, label="Sales", zorder=1)

    # Labelling figure.
    title_string = create_title_string(features, seq_length, epochs_no, learning, \
        loss_func, batch, neurons, dropout)
    xlabel_string = create_xlabel_string(total_performance['total_mae'], \
        total_performance['total_mape'], total_performance['total_mse'], \
        total_performance['total_r2'])
    plt.suptitle(f"{dataset.capitalize()} dataset predictions compared to sales for all articles", fontsize=14)
    plt.title(title_string, fontsize=12)
    plt.legend()
    plt.xlabel(xlabel_string, fontsize=12)
    plt.ylabel("Sales", fontsize=12)

    # Adjusting layout.
    plt.ylim(top=20000)
    plt.grid()
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.tight_layout(pad=0.6)
    
    # Saving plot to pdf.
    save_string = f"models/{model_path}{model_name}/{model_name}_{dataset}_total_plot.pdf"
    plt.savefig(save_string, format='pdf')
    plt.close()


# Plots actual sales compared to predictions for top and bottom 5 individual articles.
# Model path specifices subdirectory of models/ and is entered as string: "model_path/".
# Neurons are entered per layer as integers in a list.
def plot_article_predictions(article_dict, predictions_by_article, article_stats_sorted, \
    seq_length, epochs_no, loss_func, batch, neurons, learning, features, dropout, model_name, \
    dataset, model_path=""):

    # stats = [article number, mape, mse, mae, r2]
    # [0] = article number
    # [1] = mean absolute percentage error
    # [2] = mean squared error
    # [3] = mean absolute error
    # [4] = r squared

    # List of titles for plots.
    titles = ["best", "2nd best", "3rd best", "4th best", "5th best", "5th worst", \
        "4th worst", "3rd worst", "2nd worst", "worst"]
    
    # Selecting starting point based on dataset.
    if dataset == "test":
        # Setting start of test set.
        start_point = dt.date(2024, 3, 25)
    else:
        # Setting start of training set.
        start_point = dt.date(2022, 5, 23)

    # Creating list of dates for plotting.
    start_date = start_point + dt.timedelta(days=seq_length)
    dates = [start_date + dt.timedelta(days=i) for i in range(len(next(iter(predictions_by_article.values()))['actual_sale']))]

    # Plotting 5 best and 5 worst articles.
    for i in chain(range(0, 5), range(-5, 0)):
        article_no = article_stats_sorted[i][0]
        article_name = article_dict[article_no]['name']

        # Setting plot size.
        plt.figure(figsize=(10,8))

        # Plotting predictions vs. sales for individual article.
        plt.plot(dates, predictions_by_article[article_no]['prediction'], \
            label="Predictions", linestyle=(5, (10, 3)), zorder=2)
        plt.plot(dates, predictions_by_article[article_no]['actual_sale'], \
            label="Sales", zorder=1)
        
        # Labelling figure.
        title_string = create_title_string(features, seq_length, epochs_no, learning, \
            loss_func, batch, neurons, dropout)
        xlabel_string = create_xlabel_string(article_stats_sorted[i][3], article_stats_sorted[i][1], \
            article_stats_sorted[i][2], article_stats_sorted[i][4])
        plt.suptitle(f"""{dataset.capitalize()} dataset predictions compared to sales for {titles[i]} performing article
{article_no}: {article_name}""", fontsize=14)        
        plt.title(title_string, fontsize=12)        
        plt.legend()        
        plt.xlabel(xlabel_string, fontsize=12)        
        plt.ylabel("Sales", fontsize=12)

        # Adjusting layout.
        plt.grid()
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
        plt.tight_layout(pad=0.6)
        
        # Saving plot to pdf.
        save_string = f"models/{model_path}{model_name}/{model_name}_{dataset}_rank{i}_article_{article_no}_plot.pdf"
        plt.savefig(save_string, format='pdf')
        plt.close()


# Plots actual sales compared to predictions for individual articles.
# Article numbers are entered as strings in a list.
# ["article1", "article2", ...]
# Model path specifices subdirectory of models/ and is entered as string: "model_path/".
# Neurons are entered per layer as integers in a list.
def plot_specific_article_predictions(article_nos, article_dict, predictions_by_article, \
    seq_length, epochs_no, loss_func, batch, neurons, learning, features, dropout, model_name, \
    dataset, model_path=""):

    # stats = [article number, mape, mse, mae, r2]
    # [0] = article number
    # [1] = mean absolute percentage error
    # [2] = mean squared error
    # [3] = mean absolute error
    # [4] = r squared
    
    # Selecting starting point based on dataset.
    if dataset == "test":
        # Setting start of test set.
        start_point = dt.date(2024, 3, 25)
    else:
        # Setting start of training set.
        start_point = dt.date(2022, 5, 23)

    # Creating list of dates for plotting.
    start_date = start_point + dt.timedelta(days=seq_length)
    dates = [start_date + dt.timedelta(days=i) for i in range(len(next(iter(predictions_by_article.values()))['actual_sale']))]
 
    # Looping through all article numbers.
    for article_no in article_nos:

        # Retrieving article name.
        article_name = article_dict[article_no]['name']

        # Calculating statistics.
        article_sales = predictions_by_article[article_no]['actual_sale']
        article_predictions = predictions_by_article[article_no]['prediction']
        mape = mean_absolute_percentage_error(article_sales, article_predictions)
        mse = mean_squared_error(article_sales, article_predictions)
        mae = mean_absolute_error(article_sales, article_predictions)
        r2 = r2_score(article_sales, article_predictions)

        # Setting plot size.
        plt.figure(figsize=(10,8))

        # Plotting predictions vs. sales for individual article.
        plt.plot(dates, article_predictions, \
            label="Predictions", linestyle=(5, (10, 3)), zorder=2)
        plt.plot(dates, article_sales, \
            label="Sales", zorder=1)
        
        # Labelling figure.
        title_string = create_title_string(features, seq_length, epochs_no, learning, \
            loss_func, batch, neurons, dropout)
        xlabel_string = create_xlabel_string(mae, mape, mse, r2)
        plt.suptitle(f"{dataset.capitalize()} dataset predictions for {article_no}: {article_name}", fontsize=14)
        plt.title(title_string, fontsize=12)
        plt.legend()
        plt.xlabel(xlabel_string, fontsize=12)
        plt.ylabel("Sales", fontsize=12)

        # Adjusting layout.
        plt.grid()
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
        plt.tight_layout(pad=0.6)
        
        # Saving plot to pdf.
        save_string = f"models/{model_path}{model_name}/{model_name}_{dataset}_article_{article_no}_{article_name.replace(' ', '-')}_plot.pdf"
        plt.savefig(save_string, format='pdf')
        plt.close()


# Plotting r2 scores and RMSE for model training by number of epochs.
# R2 scores and RMSE values for test and training sets are entered as items in a list.
# The number of neurons is entered as items in a list: e.g., [176] or [176, 88].
def plot_training_stats(r2_test, r2_training, rmse_test, rmse_training, epochs, \
        features, seq_length, learning, loss_func, batch, neurons, dropout):

    # Creating string denoting span of epochs.
    epoch_string = f"{epochs[0]}-{epochs[-1]}"

    # Deriving model name from input parameters.    
    model_name = create_model_name(seq_length, epoch_string, loss_func, batch, \
    neurons, learning, features, dropout)

    # Setting plot size.
    plt.figure(figsize=(10,6))

    # Plotting r squared for test and training set.
    plt.plot(epochs, r2_test, label="Test set", zorder=2)
    plt.plot(epochs, r2_training, label="Training set", zorder=1)

    # Labelling figure.
    title_string = create_title_string(features, seq_length, epoch_string, learning, \
            loss_func, batch, neurons, dropout)
    plt.suptitle("Coefficient of determination ($R^2$) across training epochs" , fontsize=14)
    plt.title(title_string, fontsize=12)
    plt.legend()
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("$R^2$", fontsize=12)

    # Adjusting layout.
    plt.grid()
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.tight_layout(pad=0.6)
    
    # Saving plot to pdf.
    save_string = f"plots/test_results/{model_name}_r2_plot.pdf"
    plt.savefig(save_string, format='pdf')
    plt.close()

    # Setting plot size.
    plt.figure(figsize=(10,6))

    # Plotting RMSE for test and training set.
    plt.plot(epochs, rmse_test, label="Test set", zorder=2)
    plt.plot(epochs, rmse_training, label="Training set", zorder=1)

    # Labelling figure.
    title_string = create_title_string(features, seq_length, epoch_string, learning, \
            loss_func, batch, neurons, dropout)
    plt.suptitle("Root mean squared error (RMSE) across training epochs" , fontsize=14)
    plt.title(title_string, fontsize=12)
    plt.legend()
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)

    # Adjusting layout.
    plt.grid()
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=8))
    plt.tight_layout(pad=0.6)
    
    # Saving plot to pdf.
    save_string = f"plots/test_results/{model_name}_rmse_plot.pdf"
    plt.savefig(save_string, format='pdf')
    plt.close()


#####################
### Model testers ###
#####################

# Function for testing stored model. 
# Article numbers are entered as strings in a list: e.g., ["article1", "article2", ...]
# Model directory is derived from specified hyperparameters.
# The number of neurons is entered as items in a list: e.g., [176] or [176, 88].
# Model path specifices subdirectory of models/ and is entered as string: e.g., "model_path/".
def plot_articles_from_saved_model(article_nos, seq_length, epochs_no, loss_func, \
    batch, neurons, learning, features, dropout, model_path=""):    

    # Creating string for model name.
    model_name = create_model_name(seq_length, epochs_no, loss_func, batch, neurons, \
        learning, features, dropout)

    # Loading model.
    model = keras.saving.load_model(f"models/{model_path}{model_name}/{model_name}.keras")
 
    # Creating list with all information about article.
    article_info = ex.extract_all_articles_info_hot_encoded('training_data/325_articles.csv', \
        'data/json/article_dictionary.json')

    # Creating training and test sequences.
    sequences = ex.create_dictionary_sequences(article_info, seq_length)

    # Loading article dictionary for plotting and printing.
    article_dict = ex.load_json_dictionary('data/json/article_dictionary.json')

    # Making and plotting predictions for test set.
    create_prediction_plot_articles(model, sequences['test_input'], \
        sequences['test_output'], article_nos, article_dict, seq_length, \
        epochs_no, loss_func, batch, neurons, learning, features, dropout, \
        model_name, "test", model_path)

    # Making and plotting predictions for training set.
    create_prediction_plot_articles(model, sequences['training_input'], \
        sequences['training_output'], article_nos, article_dict, seq_length, \
        epochs_no, loss_func, batch, neurons, learning, features, dropout, \
        model_name, "training", model_path)    
    

# Function for testing stored model.
# Model directory is derived from specified hyperparameters.
# The number of neurons is entered as items in a list: e.g., [176] or [176, 88].
# Model path specifices subdirectory of models/ and is entered as string: e.g., "model_path/".
def test_saved_model(seq_length, epochs_no, loss_func, batch, neurons, learning, features, \
    dropout, model_path=""):

    # Creating string for model name.
    model_name = create_model_name(seq_length, epochs_no, loss_func, batch, neurons, \
        learning, features, dropout)

    # Loading model.
    model = keras.saving.load_model(f"models/{model_path}{model_name}/{model_name}.keras")
 
    # Creating list with all information about article.
    article_info = ex.extract_all_articles_info_hot_encoded('training_data/325_articles.csv', \
        'data/json/article_dictionary.json')

    # Creating training and test sequences.
    sequences = ex.create_dictionary_sequences(article_info, seq_length)

    # Loading article dictionary for plotting and printing.
    article_dict = ex.load_json_dictionary('data/json/article_dictionary.json')

    # Creating, printing and plotting statistics for test set.
    create_prediction_statistics(model, sequences['test_input'], sequences['test_output'], \
        article_dict, seq_length, epochs_no, loss_func, batch, neurons, learning, features, \
        dropout, model_name, "test", model_path)
    
    # Creating, printing and plotting statistics for training set.
    create_prediction_statistics(model, sequences['training_input'], sequences['training_output'], \
        article_dict, seq_length, epochs_no, loss_func, batch, neurons, learning, features, dropout, \
        model_name, "training", model_path)
    

# Tests model directly during creation to create statistics and plots.
# The number of neurons is entered as items in a list: e.g., [176] or [176, 88].
# Model path specifices subdirectory of models/ and is entered as string: e.g., "model_path/".
def test_model(model, sequences, model_name, seq_length, epochs_no, loss_func, batch, \
    neurons, learning, features, dropout, model_path):

    # Loading article dictionary for plotting and printing.
    article_dict = ex.load_json_dictionary('data/json/article_dictionary.json')

    # Creating, printing and plotting statistics for test set.
    create_prediction_statistics(model, sequences['test_input'], sequences['test_output'], \
        article_dict, seq_length, epochs_no, loss_func, batch, neurons, learning, features, \
             dropout, model_name, "test", model_path)
    
    # Creating, printing and plotting statistics for training set.
    create_prediction_statistics(model, sequences['training_input'], sequences['training_output'], \
        article_dict, seq_length, epochs_no, loss_func, batch, neurons, learning, features, \
        dropout, model_name, "training", model_path)