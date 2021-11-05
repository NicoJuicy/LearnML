# Plots and Graphs

## Bokeh and Cufflinks

In this section, we discuss two visualization libraries  `pandas_bokeh` and `cufflinks` to create plotly and bokeh charts using the basic pandas plotting syntax. 

### Import the Dataset

```py
  # Reading in the data
  df = pd.read_csv('NIFTY_data_2020.csv',parse_dates=["Date"],index_col='Date')

  # resample/aggregate the data by month-end
  df_resample = nifty_data.resample(rule = 'M').mean()
```

### Plotting using Pandas

### Plotting using Pandas-Bokeh

```py
  import pandas as pd
  import pandas_bokeh as pb
  
  # embedding plots in Jupyter Notebooks
  pb.output_notebook() 
  
  # export plots as HTML
  pb.output_file(filename) 
```

```py
  df.plot_bokeh(kind='line')
  df.plot_bokeh.line()  # same thing
  
  # scatter plot
  df.plot_bokeh.scatter(x='NIFTY FMCG index', y='NIFTY Bank index')
  
  # histogram
  df[['NIFTY FMCG index','NIFTY Bank index']].plot_bokeh(kind='hist', bins=30)

  # bar plot df_resample.plot_bokeh(kind='bar',figsize=(10,6))
```


### Plotting using Cufflinks

Cufflinks is an independent third-party wrapper library around Plotly that is more versatile, has more features, and has an API similar to pandas plotting. 

```py
import pandas as pd
import cufflinks as cf

# making all charts public and setting a global theme
from IPython.display import display,HTML

cf.set_config_file(sharing='public',theme='white',offline=True)
```

```py
  df.iplot(kind='line')
  
  # scatter plot
  df.iplot(kind='scatter',x='NIFTY FMCG index', y='NIFTY Bank index',mode='markers')
  
  # histogram
  df[['NIFTY FMCG index','NIFTY Bank index']].iplot(kind='hist', bins=30)
  
  # bar plot
  df_resample.iplot(kind='bar')
```


## Seaborn

### Import Dataset

Here we load the stock prices of Apple, Microsoft, Google, and Moderna between the given start and end dates.

```py
  import pandas as pd
  import pandas_datareader as pdr
  import seaborn as sns
  sns.set(style="darkgrid")
  
start = '2020-1-1'
end = '2021-6-30'
source = 'yahoo'
stocks = pd.DataFrame(columns=["Date","Close","Volume","Stock"])
stock_list = ["AAPL","IBM","MSFT","MRNA"]
for stock in stock_list:
    df = pdr.data.DataReader(stock, start=start ,end=end, 
                         data_source=source).reset_index()
    df["Stock"] = stock
    df = df[["Date","Close","Volume","Stock"]]
    stocks = pd.concat([stocks, df], ignore_index=True)
    stocks.head()
```

### Line Plot

We can use the relplot or lineplot functions of Seaborn to create line plots. 

The `relplot` function is a figure-level interface for drawing relational plots including line plot and scatter plot. 

```
  sns.relplot(
    data=stocks[stocks.Stock == "AAPL"], 
    x="Date", y="Close", 
    kind="line",
    height=5, aspect=2 
    )

  # increase font size of the axis titles and legend
  sns.set(font_scale=1.5)
  
  # plot all stocks
  sns.relplot(
    data=stocks, 
    x="Date", y="Close", hue="Stock", 
    height=5, aspect=2, 
    kind="line",
    palette="cool"
    ).set(
      title="Stock Prices", 
      ylabel="Closing Price",
      xlabel=None
    )

  # create line plot for each stock using row and/or col
  sns.relplot(
    data=stocks, x="Date", y="Close", 
    row="Stock",
    height=3, aspect=3.5,
    kind="line"
    )
```

## Seaborn Tips

### Changing the Font Size in Seaborn

1. Set_theme function

<img width=600 src="https://miro.medium.com/max/2100/1*z3Bu9_mGcoVNt7ueVwz5nQ.png" />

2. Axis level functions

<img width=600 src="https://miro.medium.com/max/2100/1*HXJRNs84Wc23NnmJ9OomKA.png" />

3. Set_axis_labels function

<img width=600 src="https://miro.medium.com/max/2100/1*ELwHL4YC5ombql6yJKLdGA.png" />

4. Matplotlib functions

<img width=600 src="https://miro.medium.com/max/2100/1*WgJOamXxhwolExnNZEzl-g.png" />


Seaborn allows for creating the common plots with just 3 functions:

- Relplot: Used for creating relational plots
- Displot: Used for creating distributions plots
- Catplot: Used for creating categorical plots


## Data Visualization Packages

In this section, we discuss three visualization python packages to help with data science activities. 

### AutoViz

AutoViz is an open-source visualization package under the AutoViML package library designed to automate many data scientists’ works. Many of the projects were quick and straightforward but undoubtedly helpful, including AutoViz.

AutoViz is a one-liner code visualization package that would automatically produce data visualization. 


### Missingno

missingno is a package designed to visualize your missing data. 

This package provides an easy-to-use insightful one-liner code to interpret the missing data and shows the missing data relationship between features. 

### Yellowbricks

Yellowbrick is a library to visualize the machine learning model process.

Yellowbrick is an open-source package to visualize and work as diagnostic tools that build on top of Scikit-Learn. 

Yellowbrick was developed to help the model selection process using various visualization APIs that extended from Scikit-Learn APIs.


## References

[Get Interactive Plots Directly With Pandas](https://www.kdnuggets.com/get-interactive-plots-directly-with-pandas.html/)

[7 Examples to Master Line Plots With Python Seaborn](https://towardsdatascience.com/7-examples-to-master-line-plots-with-python-seaborn-42d8aaa383a9?gi=9da22d442565)

[The Easiest Way to Make Beautiful Interactive Visualizations With Pandas using Cufflinks](https://towardsdatascience.com/the-easiest-way-to-make-beautiful-interactive-visualizations-with-pandas-cdf6d5e91757)

[4 simple tips for plotting multiple graphs in Python](https://towardsdatascience.com/4-simple-tips-for-plotting-multiple-graphs-in-python-38df2112965c)

[Top 3 Visualization Python Packages to Help Your Data Science Activities](https://towardsdatascience.com/top-3-visualization-python-packages-to-help-your-data-science-activities-168e22178e53)

[4 Different Methods for Changing the Font Size in Python Seaborn](https://sonery.medium.com/4-different-methods-for-changing-the-font-size-in-python-seaborn-fd5600592242)

[3 Seaborn Functions That Cover All Your Visualization Tasks](https://towardsdatascience.com/3-seaborn-functions-that-cover-almost-all-your-visualization-tasks-793f76510ac3)
