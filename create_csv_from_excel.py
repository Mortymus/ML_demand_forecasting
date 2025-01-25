import extraction as ex

if __name__ == "__main__":

    # Extracting sheets from excel files to separate csv files.
    ex.extract_sales_to_csv('data/excel/sales_230522-270324.xlsx')
    ex.extract_sales_to_csv('data/excel/sales_280324-080924.xlsx')
    