# Economic and Environmental Potential of Machine Learning in Demand Forecasting
### Bachelor's thesis for the Department of Electrical Engineering and Computer Science, University of Stavanger

#### Abstract
Demand forecasting precision is essential for suppliers of perishable items to be able to meet customer demand and avoid over-stocking goods that will inevitably spoil. Accurately predicting demand enables optimal logistical organization that maximize sales and minimize waste, allowing for increased profitability and decreased environmental impact. Machine learning have the potential to improve demand forecasting accuracy in comparison with traditional methods, with a number of established approaches documented to increase precision. This bachelor thesis will implement an LSTM network to create a demand forecasting model for perishable items supplied from a regional distribution center, detailing the associated challenges and strategies for handling them. Data quality and cleaning is essential for successful model implementation, along with careful feature engineering and optimization of hyperparameters. Access to computational resources is necessary to determine optimal model configuration by testing large numbers of model architectures. Combinations of hyperparameters are systematically tested to create an initial model with selected features. Feature relevancy is tested through a series of experiments on the initial model, forming the foundation for further model optimization. Evaluation of model architecture and performance conclude further dataset and feature improvements are necessary to reach an acceptable level of demand forecasting precision, pointing to features describing the availability and type of items as intriguing avenues of future research.

# Instructions to Compile and Run System

The project has been stored using Git LFS due to storage of models exceeding 100MB. It is recommended to have Git LFS installed to be able to clone the repository correctly.

[https://git-lfs.com/](https://git-lfs.com/)

### Model folders

The Github repository contains a number of stored LSTM models in the folder named `models`. Models are stored in subfolders denoting their roles in conducted experiments, with each network stored in an individual folder named by its hyperparameters. Network folders contain a Keras model file along with plots and text files with statistics. The table below links experiments from specified sections to their associated model folders.

| **Section** | **Experiment**                     | **Folder**                          |
|-------------|-------------------------------------|--------------------------------------|
| 4.2.1   | Creating 54 initial models         | `models/initial_models`             |
| 4.2.2   | Testing relevancy of features      | `models/feature_testing`            |
| 4.2.3   | Initial model, 1-20 epochs         | `models/initial_model_20epochs`     |
| 4.2.3   | Creating 160 optimized models      | `models/optimized_models`           |
| 4.2.3   | Optimized model, 10-100 epochs     | `models/optimized_model_10-100epochs` |
| 4.2.3   | Optimized model, 50-300 epochs     | `models/optimized_model_50-300epochs` |
| 4.2.3   | Optimized model, 1-20 epochs       | `models/optimized_model_1-20epochs` |
| 4.2.3   | Stacking optimized model           | `models/optimized_model_stacking`   |
| 4.2.3   | Revising missing values            | `models/revised_missing_values`     |
| 4.2.3   | Revising article reduction         | `models/revised_article_reduction`  |

_Overview of experimental sections and folders with the associated LSTM models._

### Codebase

The codebase consists of three central files:

- `extraction.py`
- `neural_network_builder.py`
- `model_tester.py`

The central files provide support functions for each other, which can be detangled by following references in specific functions. The intention is to provide descriptive function names to make the code easy to read, with the explicit code available in the support functions. Users that are familiarized with the code can switch between datasets by updating functions, but this can be cumbersome and requires specific knowledge. The uninitiated user encounters a set of functions that is specifically geared towards using a dataset with 325 articles where samples have a total of 351 features.

The codebase includes a set of scripts:

- `create_all_articles_sales_plots.py`
- `create_article_dictionary_lookup.py`
- `create_article_dictionary.py`
- `create_csv_from_excel.py`
- `create_csv_sales_sorted_by_article_and_date.py`
- `create_initial_files.py`
- `create_lstm_network.py`
- `create_lstm_plots.py`
- `create_lstm_test.py`
- `create_zero_sale_dictionary.py`


All Python files beginning with the word "create" are scripts that draw on functions from the codebase. The code is dependent on csv files and json dictionaries that have been created during the cleaning and operationalization of the dataset. The initial process of creating necessary csv and json files can be largely recreated, but is dependent on `data/json/zero_sale_dictionary.json`. Running the script  `create_initial_files.py` will recreate all other necessary files.

Users can easily create new LSTM models using the `create_model_sequences()` and `create_lstm_neural_network()` functions from `neural_network_builder.py`. Use the script `create_lstm_network.py` for an example of how to run the code. Models are stored in directories named by their hyperparameters. Users can create subfolders to store models by using the `model_path` parameter as demonstrated in the example script.

Users can use a saved model to plot predictions for specific articles by running the `plot_articles_from_saved_model()` function from `model_tester.py`. Use the script `create\_lstm\_plots.py` for an example of how to run the code. Plots and prints will be stored to the model directory. Paths for saved models are derived from the specified hyperparameters. Users can specify which subfolder to find the model by using the `model\_path` parameter as demonstrated in the example script.

Users can run tests on saved models by using the `test_saved_model()` function from `model\_tester.py`, producing the plots and prints that are normally produced when the model is created. Run the script `create_lstm_test.py` for an example of how to run the code. Paths for saved models are derived from the specified hyperparameters. Users can specify which subfolder to find the model by using the `model_path` parameter as demonstrated in the example script.

`create_all_articles_sales_plots.py` creates plots of all article sales, storing each plot to an individual pdf. The script also merges all the pdf files into a single file to make viewing them easier. Sales plots are stored in the folder `plots/article_sales`.

The remaining scripts are used for intializing the necessary files for the project, and can be found in `create_initial_files.py`.

### Dataset

The folder `data/excel` contains the original dataset supplied by Coop Norge SA. Minor adjustments were made to copies of the Excel files, which were then saved under new names to simplify processing.

Extracted Excel sheets are stored in csv format in the folder `data/csv`. The dataset was derived from these files and stored in the `training_data` folder. There are currently 4 different files with varying numbers of articles, resulting from different iterations of dataset cleaning.

## Python environment

The project was developed on a Windows 11 machine using Python 3.11.9. Apologies for any unforeseen errors for those using a different operating system.

The following Python packages are required:

| Package                      | Version   |
|------------------------------|-----------|
| absl-py                      | 2.1.0     |
| asttokens                    | 2.4.1     |
| astunparse                   | 1.6.3     |
| certifi                      | 2024.8.30 |
| charset-normalizer           | 3.3.2     |
| colorama                     | 0.4.6     |
| comm                         | 0.2.2     |
| contourpy                    | 1.3.0     |
| cycler                       | 0.12.1    |
| debugpy                      | 1.8.5     |
| decorator                    | 5.1.1     |
| et-xmlfile                   | 1.1.0     |
| executing                    | 2.1.0     |
| flatbuffers                  | 24.3.25   |
| fonttools                    | 4.53.1    |
| gast                         | 0.6.0     |
| google-pasta                 | 0.2.0     |
| grpcio                       | 1.66.1    |
| h5py                         | 3.11.0    |
| holidays                     | 0.56      |
| idna                         | 3.8       |
| ipykernel                    | 6.29.5    |
| ipython                      | 8.27.0    |
| jedi                         | 0.19.1    |
| joblib                       | 1.4.2     |
| jupyter_client               | 8.6.2     |
| jupyter_core                 | 5.7.2     |
| keras                        | 3.5.0     |
| kiwisolver                   | 1.4.7     |
| libclang                     | 18.1.1    |
| Markdown                     | 3.7       |
| markdown-it-py               | 3.0.0     |
| MarkupSafe                   | 2.1.5     |
| matplotlib                   | 3.9.2     |
| matplotlib-inline            | 0.1.7     |
| mdurl                        | 0.1.2     |
| ml-dtypes                    | 0.4.0     |
| namex                        | 0.0.8     |
| nest-asyncio                 | 1.6.0     |
| numpy                        | 1.26.4    |
| openpyxl                     | 3.1.5     |
| opt-einsum                   | 3.3.0     |
| optree                       | 0.12.1    |
| packaging                    | 24.1      |
| pandas                       | 2.2.2     |
| parso                        | 0.8.4     |
| pillow                       | 10.4.0    |
| pip                          | 24.2      |
| platformdirs                 | 4.3.3     |
| prompt_toolkit               | 3.0.47    |
| protobuf                     | 4.25.4    |
| psutil                       | 6.0.0     |
| pure_eval                    | 0.2.3     |
| Pygments                     | 2.18.0    |
| pyparsing                    | 3.1.4     |
| pypdf                        | 4.3.1     |
| python-dateutil              | 2.9.0.post0 |
| pytz                         | 2024.1    |
| pywin32                      | 306       |
| pyzmq                        | 26.2.0    |
| requests                     | 2.32.3    |
| rich                         | 13.8.1    |
| scikit-learn                 | 1.5.1     |
| scipy                        | 1.14.1    |
| setuptools                   | 65.5.0    |
| six                          | 1.16.0    |
| stack-data                   | 0.6.3     |
| tensorboard                  | 2.17.1    |
| tensorboard-data-server      | 0.7.2     |
| tensorflow                   | 2.17.0    |
| tensorflow-intel             | 2.17.0    |
| tensorflow-io-gcs-filesystem | 0.31.0    |
| termcolor                    | 2.4.0     |
| threadpoolctl                | 3.5.0     |
| tornado                      | 6.4.1     |
| traitlets                    | 5.14.3    |
| typing_extensions            | 4.12.2    |
| tzdata                       | 2024.1    |
| urllib3                      | 2.2.2     |
| wcwidth                      | 0.2.13    |
| Werkzeug                     | 3.0.4     |
| wheel                        | 0.44.0    |
| wrapt                        | 1.16.0    |