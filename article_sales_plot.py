import datetime
import extraction as ex
import math
import matplotlib.pyplot as plt
import os
from pypdf import PdfMerger

# Creating moving average filter.
def moving_average(sales):

    # Creating 30 day moving average filter to see general trend.
    filter = []
    window = 30
    sales_sum = 0

    # Looping through indexes of sales list.
    for index in range(len(sales)):
        
        # Adding sale to total sum.
        sales_sum += sales[index]

        # Subtracting sale outside of window if applicable.
        if index + 1 >= window:
            sales_sum -= sales[index-window]

        # Calculating moving average.
        avg = math.ceil(sales_sum/(min(index + 1, window)))

        # Appending moving average to filter list.
        filter.append(avg)
    
    # Returning list with filter.
    return filter


# Plotting specific article sales.
def plot_article_sales(sales, dates, filter, article_no, article_name):
    
    # Plotting sales data.
    plt.figure(figsize=(10,6))
    plt.plot(dates, sales, label="Sales", zorder=1)
    plt.plot(dates, filter, label="Moving average filter", zorder=2)
    plt.xlim(dates[0], dates[-1])
    plt.ylim(min(min(sales), -10), max(max(sales), 10))
    plt.suptitle(f"Article number {article_no}: {article_name}", fontsize=14)
    plt.title("Sales compared to moving average filter with 30 day window", fontsize=12)
    plt.legend()
    plt.xlabel("Dates", fontsize=12)
    plt.ylabel("Sales", fontsize=12)
    plt.grid()
    
    # Saving plot to pdf.
    save_string = f"plots/article_sales/{str(article_name.replace(' ', '_').replace('/', '-'))}_{article_no}_plot.pdf"
    plt.savefig(save_string, format='pdf')

    # Closing plot.
    plt.close()


# Plotting all article sales.
def plot_all_article_sales():    

    # Creating article dictionary with sales information.
    # article_info['article number'] = [...], [...], [...], ...
    # [0] = article_number
    # [1] = year
    # [2] = month
    # [3] = day
    # [4] = holiday
    # [5] = sale
    article_info = ex.extract_article_dictionary("training_data/402_articles.csv")

    # Creating list with all dates.
    dates = []

    # Looping through all values of one key to retrieve dates.
    for line in article_info[next(iter(article_info))]:
        
        # Appending dates from one key that are universal for all articles.
        dates.append(datetime.date(line[1], line[2], line[3]))

    # Creating article dictionary.
    article_dict = ex.load_json_dictionary('data/json/article_dictionary.json')

    # Extracting sales and plotting information for each article.
    for article_no in article_info:

        # Creating list for sale values.
        sales = []

        # Extracting sales.
        for line in article_info[article_no]:
            sales.append(line[5])        

        # Calculating moving average of sales.
        filter = moving_average(sales)

        # Retrieving article name.
        article_name = article_dict[str(article_no)]['name']

        # Creating article sale plot.
        plot_article_sales(sales, dates, filter, article_no, article_name)


# Merging the pdfs of all articles into one.
def merge_pdfs(directory):

    # Initializing pdf merger.
    merger = PdfMerger()

    # Listing all pdf files in directory.
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.pdf'):
            merger.append(f"{directory}/{filename}")
    
    # Saving merged pdf file.
    merger.write(f"{directory}/1_all_articles_sales_plot.pdf")
    merger.close()