# Contains all functions related to extraction 
# and operationalization of the dataset.

import csv
import datetime
import holidays
import json
import numpy as np
import pandas

###############
### Classes ###
###############

# Campaign class.
class Campaign:

    # Start and stop of campaigns for specific store.
    def __init__(self, article_no, start_date, stop_date, store_no):
        self.start_date = start_date
        self.stop_date = stop_date
        self.store_no = store_no
        self.article_no = article_no


    # Standard print of object.
    def __str__(self):
        return f"{self.article_no}: {self.start_date}, {self.stop_date}, {self.store_no}"
    

    # Checking if campaign is active for specific date.
    def check_campaign(self, date, campaign_list):
        if self.start_date <= date <= self.stop_date \
            and self.store_no not in campaign_list:
                campaign_list.append(self.store_no)


#########################
### Support functions ###
#########################

# Creating json dictionary.
def create_json_file(dictionary, json_path):
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(dictionary, json_file, ensure_ascii=False, indent=4)


# Loading json dictionary.
def load_json_dictionary(json_path):
    
    # Loading article dictionary from json file.
    with open(json_path, 'r', encoding='utf-8') as json_file:
        dictionary = json.load(json_file)

    # Returning dictionary.
    return dictionary


# Creating string from list items.
def list_item_string(list):
    
    # Creating empty string for list items.
    list_string = ""
    
    # Adding each item to string.
    for item in list:

        # Adding dash between items.
        if len(list_string) > 0:
            list_string += "-"

        list_string += str(item)
    
    # Returning string with list items.
    return list_string


#####################################
### Excel file conversions to csv ###
#####################################

# Extract output data from excel to csv.
def extract_sales_to_csv(sales_file):

    # Converting excel file to pandas.ExcelFile object.
    excel_file = pandas.ExcelFile(sales_file)

    # Retrieving sheet names from object.
    sheets = excel_file.sheet_names

    # Exporting each sheet of excel file as csv file with delimiter ';'.
    for sheet in sheets:
        excel_sheet = pandas.read_excel(sales_file, sheet_name=sheet)
        ssv_path = "data/csv/" + sheet.replace(".","_") + ".csv"
        excel_sheet.to_csv(ssv_path, sep=";", index=False)


#####################################
### Excel file extraction to json ###
#####################################

# Extract article number and name to json dictionary.
def extract_article_number_and_name(excel_path, json_path):

    # Converting excel file to pandas.DataFrame object.
    excel_file = pandas.read_excel(excel_path, header=None)

    # Creating dictionary for article numbers and names.
    articles = {}

    # Loading dictionary with articles that are excluded due to missing sales.
    zero_sale_dict = load_json_dictionary("data/json/zero_sale_dictionary.json")

    # Storing name and translated article number for each article.
    counter = 1
    for _, row in excel_file.iterrows():
        
        # Using article number as key in dictionary.
        article_index = str(row[0])

        # Ensuring repeating articles is only counted once.
        if article_index not in articles and article_index not in zero_sale_dict:
            articles[article_index] = {
                'name': row[1], 
                'number': counter, 
            }
            counter += 1

    # Saving article dictionary to json file.
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(articles, json_file, ensure_ascii=False, indent=4)


############################################
### Article number and name translations ###
############################################

# Translating article number to name.
def article_number_to_name(article_number, json_path="data/json/article_dictionary.json"):
    
    # Loading article dictionary from json file.    
    article_dictionary = load_json_dictionary(json_path)

    # Returning name for given article number.
    return article_dictionary[str(article_number)]['name']


# Translating article name to number.
def article_name_to_number(article_name, json_path="data/json/article_dictionary.json"):

    # Loading article dictionary from json file.    
    article_dictionary = load_json_dictionary(json_path)

    # Creating list for articles containing given string.
    articles = {}

    # Adding articles to list.
    for key, value in article_dictionary.items():
        if article_name.lower() in value['name'].lower():
            articles[key] = value
    
    # Returning list of articles.
    return articles


# Creating reverse article number lookup based on one-hot-encoded value.
def create_article_dictionary_lookup(article_dict_path, new_json_path):

    # Loading article dictionary.
    article_dict = load_json_dictionary(article_dict_path)

    # Creating reverse lookup article dictionary.
    article_dictionary_lookup = {}

    # Gathering article information in new dictionary with 
    # one-hot encoding value as key.
    for article in article_dict:
        article_dictionary_lookup[article_dict[article]['number']] = {
            'name': article_dict[article]['name'], 
            'article_number': article
        }
    
    # Creating article dictionary lookup json file.
    with open(new_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(article_dictionary_lookup, json_file, ensure_ascii=False, indent=4)


########################################
### Data extraction from excel files ###
########################################

# Extract campaigns to dictionary.
def extract_campaigns():

    # Campaign files for data extraction.
    campaign_files = ['data/excel/campaigns_230522-270324.xlsx',
                  'data/excel/campaigns_280324-080924.xlsx']

    # Creating campaign dictionary.
    campaign_dict = {}

    # Extracting campaigns from each campaign file.
    for campaign_file in campaign_files:

        df = pandas.read_excel(campaign_file,
            usecols=['Distribusjonskanal', 'Tilbudsstart', 'Tilbudsslutt', 'Produkt-ID'])

        for _, row in df.iterrows():

            # Extracting article number.
            article_no = str(row['Produkt-ID'])
            
            # Creating article number key if not in dictionary.
            if article_no not in campaign_dict:
                campaign_dict[article_no] = []
            
            # Extracting start and stop date of campaign.
            start_date = row['Tilbudsstart'].date()
            stop_date = row['Tilbudsslutt'].date()

            # Extracting store number.
            store_no = int(row['Distribusjonskanal'][-3:-1])

            # Creating campaign object.
            campaign = Campaign(article_no, start_date, stop_date, store_no)
            
            # Adding campaign to dictionary.
            campaign_dict[article_no].append(campaign)

    return campaign_dict


######################################
### Data extraction from csv files ###
######################################

# Extracting dates from header.
def extract_dates_from_header(csv_file):

    # Creating list for saving dates.
    dates = []

    # Opening file to extract dates.
    with open(csv_file) as data:
        
        # Extracting dates from header.
        header_dates = next(data).split(';')[2:]

        # Adding dates to list.
        for date in header_dates:
            year = int(date.strip('\n')[-4:])
            month = int(date[3:5])
            day = int(date[0:2])
            dates.append(datetime.date(year, month, day))

    # Returning extracted dates.
    return dates


# Extracting information for all articles from csv files and saving to single csv file.
def extract_article_info_to_single_csv(csv_files):
    
    # Dictionary for storing extracted data by article.
    article_data = {}

    # Creating overview of Norwegian Holidays from 2022 to 2025.
    years = range(2022, 2026)
    holidays_no = holidays.NO(years=years, language='no')

    # Adding Christmas Eve and New Year's Eve.
    for year in years:
        holidays_no.append({datetime.date(year, 12, 24): "Julaften"})
        holidays_no.append({datetime.date(year, 12, 31): "Nyttårsaften"})

    # Loading dictionary with excluded articles.
    zero_sale_dictionary = load_json_dictionary('data/json/zero_sale_dictionary.json')

    # Looping through csv files.
    for file in csv_files:

        # Extracting dates from csv file.
        dates = extract_dates_from_header(file)

        # Opening file to extract output.
        with open(file, 'r') as data:

            # Skipping header.
            data.readline()

            # Extracting output from individual lines.
            for line in data:
            
                # Splitting line items.
                line = line.split(';')

                # Extracting article number.
                article_no = line[1]

                # Checking if article is excluded.
                if article_no not in zero_sale_dictionary:

                    # Creating article key if not present.
                    if article_no not in article_data:
                        article_data[article_no] = []

                    # Commparing dates to number of output.
                    if len(dates) == len(line[2:]):
                        
                        # Extracting output for specific dates.
                        for date, sale in zip(dates, line[2:]):

                                # List for article information for specific date.
                                # [0] = article number
                                # [1] = year
                                # [2] = month
                                # [3] = day
                                # [4] = holiday
                                # [5] = sale
                                article_line = []

                                # Adding article number.
                                article_line.append(article_no)
            
                                # Adding timestamp information.
                                article_line.append(date.year)
                                article_line.append(date.month)
                                article_line.append(date.day)

                                # Adding holiday information.
                                if date in holidays_no:                                    
                                    article_line.append(1)
                                else:
                                    article_line.append(0)

                                # Adding sale for date.
                                article_line.append(float(sale))

                                # Adding complete line of item data for specific date.
                                article_data[article_no].append(article_line)   
                    
                    # Exception if mismatch between dates and output.
                    else:
                        raise Exception("Unequal number of dates and entries!")        

    # Creating file to store all articles.
    with open('training_data/325_articles.csv', 'w', newline="") as total_articles:
        
        # Creating writer for article information.
        total_writer = csv.writer(total_articles, delimiter=';')

        # Looping through keys of article dictionary
        for key in sorted(article_data.keys(), key=int):
            
            # Writing article information to file.
            total_writer.writerows(article_data[key])


# Extracting article dictionary from single csv file.
def extract_article_dictionary(csv_file):

    # Creating article dictionary.
    article_info = {}

    # Opening csv file.
    with open(csv_file, 'r') as file:

        # Looping through csv file.
        # csv line format:
        # [0] = article number
        # [1] = year
        # [2] = month
        # [3] = day
        # [4] = holiday
        # [5] = sale
        for line in file:

            # Splitting line into data instances.
            line = line.split(';')

            # Creating list for article information.
            article_line = []

            # Extracting and appending article number.
            article_no = int(line[0])
            article_line.append(article_no)

            # Creating dictionary key if not present.
            if article_no not in article_info:
                article_info[article_no] = []

            # Extracting and appending year.
            article_line.append(int(line[1]))

            # Extracting and appending month.
            article_line.append(int(line[2]))

            # Extracting and appending day.
            article_line.append(int(line[3]))
            
            # Extracting and appending holiday info.
            article_line.append(int(line[4]))

            # Extracting and appending sale info.
            article_line.append(float(line[5][:-2]))

            # Appending article line to dictionary.
            article_info[article_no].append(article_line)
    
    # Returning dictionary with all articles.
    return article_info


# Create dictionary of all articles that have total sales of zero or below.
def extract_zero_sale_articles(json_path):

    # Creating dictionary with all articles.
    article_info = extract_article_dictionary('training_data/429_articles.csv')

    # Creating dictionary to store articles with zero sales.
    zero_sale_articles = {}

    # Looping through all articles.
    for article_no in article_info:

        # Setting total sales to zero.
        total_sales = 0

        # Looping through all article sales to calculate total.
        for line in article_info[article_no][-365:]:

            # Adding sale for specific date to total.
            total_sales += line[5]

        # Adding articles with total sales of zero or below to dictionary.
        if total_sales <= 0:                        
            zero_sale_articles[article_no] = f"Total sales: {total_sales}"
    
    # Writing zero sale dictionary to json.
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(zero_sale_articles, json_file, ensure_ascii=False, indent=4)


# Create list of all dates with negative sales.
def extract_negative_sales(json_directory):

    # Creating dictionary with all sales.
    # article_info['article_no'] = [[...], [...], [...]]
    # [...] format:
    # [0] = article number
    # [1] = year
    # [2] = month
    # [3] = day
    # [4] = holiday
    # [5] = sale
    article_info = extract_article_dictionary('training_data/429_articles.csv')

    # Creating dictionary for storing dates for negative sales.
    negative_sale_dates = {}

    # Creating dictionary for storing each instance of negative sales.
    negative_sales = {}

    # Looping through all articles.
    for article_no in article_info:

        # Looping through all dates.
        for line in article_info[article_no]:

            # Finding date of negative sale.
            if line[5] < 0:
                date = datetime.date(line[1], line[2], line[3])
                date_string = date.isoformat()

                # Storing date if not already discovered.
                if date_string not in negative_sale_dates:
                    negative_sale_dates[date_string] = True
                
                # Creating key for article if non-existent.
                if article_no not in negative_sales:
                    negative_sales[article_no] = []
                
                # Storing sale and date for specific article.
                negative_sales[article_no].append(line)

    # Creating json dictionary with negative sales.
    create_json_file(negative_sales, f"{json_directory}/negative_sales_dictionary.json")

    # Creating list to sort article_numbers.
    article_numbers = []

    # Extracting article numbers from negative sales dictionary.
    for article_no in sorted(negative_sales.keys(), key = int):
        article_numbers.append(article_no)

    # Creating json file with sorted article numbers.
    create_json_file(article_numbers, f"{json_directory}/negative_sale_article_numbers_list.json")

    # Creating list to sort dates.
    negative_dates = []

    # Adding dates to list.
    for date in negative_sale_dates:
        negative_dates.append(date)

    # Sorting dates with negative sales.
    negative_dates = sorted(negative_dates)
    
    # Creating json file with sorted dates for negative sales.
    create_json_file(negative_dates, f"{json_directory}/negative_sale_dates_list.json")

    # Creating txt file with negative sale statistics.
    negative_sale_statistics(article_info, negative_sales, negative_dates)


# Print statistics about negative sales.
def negative_sale_statistics(article_info, negative_sales, negative_dates):

    # Calculating minimum, maximum and average number of negative sales per article.
    min = (100, 0)
    max = (0, 0)
    sum = 0

    # Checking each article with negative sales.
    for article_no in negative_sales:

        # Finding number of negative sales.
        no_sales = len(negative_sales[article_no])
        
        # Finding minimum number of negative sales.
        if no_sales < min[0]:
            min = (no_sales, article_no)

        # Finding maximum number of sales.
        if no_sales > max[0]:
            max = (no_sales, article_no)

        # Summarizing total number of negative sales.
        sum += no_sales

    # Finding average number of negative sales per article.
    avg = sum//len(negative_sales)

    # Printing statistics for negative sales to txt file.
    with open("data/json/negative_sale_statistics.txt", "w", encoding='utf-8') as file:

        # Retrieving article dictionary.
        article_dictionary = load_json_dictionary("data/json/article_dictionary.json")
        file.write(f"""Total number of articles: {len(article_info)}
Number of articles with negative sales: {len(negative_sales)}
First date with negative sales: {negative_dates[0]}
Last date with negative sales: {negative_dates[-1]}
Average number of negative sales: {avg}
Minimum number of sales: {min[0]}, article {min[1]}
Maximum number of sales: {max[0]}, article {max[1]}""")


# Extract info about all articles to dictionary.
# csv_file = input csv file with all articles.
# article_dict_json = path for json article dictionary.
def extract_all_articles_info_hot_encoded(csv_file, article_dict_json):
    
    # Extracting dictionary with samples sorted by article.
    # article_info['article_no'] =
    # [0] = article number
    # [1] = year
    # [2] = month
    # [3] = day
    # [4] = holiday
    # [5] = sale
    article_info = extract_article_dictionary(csv_file)

    # Loading article dictionary for number translation.
    article_dict = load_json_dictionary(article_dict_json)

    # Loading campaign dictionary.
    campaign_dict = extract_campaigns()

    # Setting index start points for different types of features.
    article_number_start_index = -1
    holiday_index = 325
    campaign_start_index = 325
    weekday_start_index = 331
    month_start_index = 338

    # Setting test index based on test date 2024-03-25.
    # Total 840 samples.
    # 80% training data = 672 samples
    # 20% test data = 168 samples
    training_index = 671

    # Creating dictionary for returnin one-hot-encoded samples.
    all_articles = {}
    all_articles['training_input'] = {}
    all_articles['training_output'] = {}
    all_articles['test_input'] = {}
    all_articles['test_output'] = {}

    # Looping through each article.
    for article in article_info:

        # Defining article number as string for compatibility 
        # with json dictionaries.        
        article_no = str(article)
        
        # Creating article keys for lists.    
        all_articles['training_input'][article_no] = []
        all_articles['training_output'][article_no] = []
        all_articles['test_input'][article_no] = []
        all_articles['test_output'][article_no] = []
        
        # Looping through samples of each article.
        for i in range(len(article_info[article])):
        
            # Defining indexes for different features.

            # Article number: [0] - [324]
            # [0] = 704197901 (1)
            # ...
            # [324] = 7110081 (325)

            # Holiday: [325]
            # [325] = Holiday (0=no, 1=yes)

            # Campaigns: [326] - [331]
            # [326] = Coop Prix (1)
            # [327] = Coop Obs (2)
            # [328] = Coop Mega (3)
            # [329] = Coop Marked (4)
            # [330] = Extra (7-2)
            # [331] = Matkroken (8-2)

            # Weekday: [332] - [338]
            # [332] = Monday (1)
            # [333] = Tuesday (2)
            # [334] = Wednesday (3)
            # [335] = Thursday (4)
            # [336] = Friday (5)
            # [337] = Saturday (6)
            # [338] = Sunday (7)
            
            # Month: [339] - [350]
            # [339] = January (1)
            # [340] = February (2)
            # [341] = March (3)
            # [342] = April (4)
            # [343] = May (5)
            # [344] = June (6)
            # [345] = July (7)
            # [346] = August (8)
            # [347] = September (9)
            # [348] = October (10)
            # [349] = November (11)
            # [350] = December (12)

            # Creating list to store article features for specific date.
            article_line = [0 for _ in range(351)]            
                      
            # Finding applicable date.
            date = datetime.date(article_info[article][i][1], \
                article_info[article][i][2], article_info[article][i][3])

            ## Adding features ##

            # One-hot encoding article number.
            article_one_hot_encoded = article_dict[article_no]['number']            
            
            # Adding one-hot encoded article number to features.
            article_line[article_number_start_index + article_one_hot_encoded] = 1            
            
            # Adding holiday information.
            if article_info[article][i][4] == 1:
                article_line[holiday_index] = 1

            # # Checking for active campaigns.
            if article_no in campaign_dict:
                
                # Creating list to store active campaigns.
                active_campaigns = []

                # Checking if campaigns for article are active.
                for campaign in campaign_dict[article_no]:
                    campaign.check_campaign(date, active_campaigns)
                
                # Adding active campaigns.
                for store in active_campaigns:
                    if store >= 7:
                        store -= 2
                    article_line[campaign_start_index+store] = 1

            # Adding weekday.            
            article_line[weekday_start_index + date.isoweekday()] = 1
            
            # Adding month.
            article_line[month_start_index + date.month] = 1

            # Checking and correcting sales for given article.
            sales = article_info[article][i][5]
            if sales < 0:
                sales = 0                     

            # Appending samples and sales to training or test set.            
            if i <= training_index:
                all_articles['training_input'][article_no].append(article_line)
                all_articles['training_output'][article_no].append(sales)            
            else:                
                all_articles['test_input'][article_no].append(article_line)
                all_articles['test_output'][article_no].append(sales)
        
    # Returning extracted training and test set.
    return all_articles


#########################
### Sequence creation ###
#########################

# Creates sequences based on sequence length from input and output dictionaries.
def create_sequences(input_dict, output_dict, seq_length):

    # Function creates sequences for input and corresponding output data.
    # input_dict = one-hot encoded input data
    # output_dict = target data.
    # seq_length = length of input and output sequences.
    
    # Creating list to hold input features and output targets.
    input_sequences = []
    output_sequences = []

    # Looping through all articles in input dictionary.
    for article in input_dict:

        # Creating sequence starting from index i in input.
        for i in range(len(input_dict[article])-seq_length):

            # Creating list with input input_sequence.
            input_sequence = input_dict[article][i:i+seq_length]                        
                        
            # Appending input sequence to list of input sequences.
            input_sequences.append(input_sequence) 
            
            # Appending single target value for input sequence to output list.
            output_sequences.append(output_dict[article][i+seq_length])                       
        
    return {'input': np.array(input_sequences), 'output': np.array(output_sequences)}


def create_dictionary_sequences(article_dict, seq_length):

    # Function creates scaled training and test sequences for a given article.
    # article_dict['input'] = input data.
    # article_dict['output] = output data.
    # seq_length = length of input and output sequences.

    # Creating training and test sequences
    training_data = create_sequences(article_dict['training_input'], \
        article_dict['training_output'], seq_length)
    test_data = create_sequences(article_dict['test_input'], \
        article_dict['test_output'], seq_length)
        
    # Creating sequences dictionary for returning training and testing data.
    sequences = {
        'training_input': training_data['input'],
        'training_output': training_data['output'],
        'test_input': test_data['input'],
        'test_output': test_data['output']
    }

    # Returning training and testing sequences.
    return sequences