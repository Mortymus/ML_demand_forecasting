import extraction as ex

if __name__ == "__main__":

    # Path for excel file with article numbers and names.
    excel_path = "data/excel/Varenummer FG DC09.xlsx"
    
    # Path to store json file.
    json_path = "data/json/article_dictionary.json"

    # Extracting dictionary with article number keys and name values.
    ex.extract_article_number_and_name(excel_path, json_path)