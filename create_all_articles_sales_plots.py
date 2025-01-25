import article_sales_plot as plot

# Creates plots of all article sales.
# Merges all the plots into single pdf.
if __name__ == "__main__":
    
    # Plotting sales for all articles.
    plot.plot_all_article_sales()

    # Merging all pdf plots into one file.
    plot.merge_pdfs('plots/article_sales')