import extraction as ex

# Script for creating a dictionary containing all articles with
# total sales equal to zero or below.

if __name__ == "__main__":

    # Setting path for storing zero sale dictionary.
    json_path = 'data/json/zero_sale_dictionary.json'

    # Creating zero sale dictionary.
    ex.extract_zero_sale_articles(json_path)