import extraction as ex

if __name__ == "__main__":
     
    # csv files containing all sales.
    # The order of the files ensures the data is sorted by date.
    csv_files = ['data/csv/23_5_22-2_10_22.csv', 'data/csv/3_10_22-2_4_23.csv', 
                'data/csv/3_4_23-1_10_23.csv', 'data/csv/2_10_23-31_3_24.csv',
                'data/csv/28_3_24-8_9_24.csv']
     
    # Extracting and storing information for each article.
    ex.extract_article_info_to_single_csv(csv_files)