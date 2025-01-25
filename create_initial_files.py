import extraction as ex

# Extracting sheets from excel files to separate csv files.
ex.extract_sales_to_csv('data/excel/sales_230522-270324.xlsx')
ex.extract_sales_to_csv('data/excel/sales_280324-080924.xlsx')

# Creating list with csv files containing all sales.
# The order of the files ensures the data is sorted by date.
csv_files = ['data/csv/23_5_22-2_10_22.csv', 'data/csv/3_10_22-2_4_23.csv', 
            'data/csv/3_4_23-1_10_23.csv', 'data/csv/2_10_23-31_3_24.csv',
            'data/csv/28_3_24-8_9_24.csv']

# Extracting and storing information for each article in a single csv file.
# Function is dependent on zero sale dictionary and csv files.
ex.extract_article_info_to_single_csv(csv_files)

# Path for excel file with article numbers and names.
excel_path = "data/excel/Varenummer FG DC09.xlsx"

# Path to store json file.
article_path = "data/json/article_dictionary.json"

# Extracting dictionary with article number keys and name values.
# Function is dependent on zero sale dictionary.
ex.extract_article_number_and_name(excel_path, article_path)

# Setting path for article dictionary.
article_dict_path = 'data/json/article_dictionary.json'

# Setting path to store lookup dictionary.
lookup_path = 'data/json/article_dictionary_lookup.json'

# Creating reverse lookup dictionary.
ex.create_article_dictionary_lookup(article_dict_path, lookup_path)