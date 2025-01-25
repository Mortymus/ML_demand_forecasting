import extraction as ex

if __name__ == "__main__":

    # Setting path for article dictionary.
    article_dict_path = 'data/json/article_dictionary.json'

    # Setting path to store new json dictionary.
    new_json_path = 'data/json/article_dictionary_lookup.json'

    # Creating reverse lookup dictionary.
    ex.create_article_dictionary_lookup(article_dict_path, new_json_path)