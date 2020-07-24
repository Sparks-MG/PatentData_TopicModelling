## dependencies

##backend
from flask import Flask
from flask import render_template, jsonify
import json
import plotly
from plotly.graph_objs import Pie, Bar, Histogram, Layout, Figure

##data analysis
import pandas as pd


app = Flask(__name__, static_url_path='/static')# static url path important for loading local files


# load the csv data
df = pd.read_csv(r'../datasets/patentFullTextData.csv', dtype= object)




# index webpage displays information about the data set
@app.route('/')
def index():

    # generate list with plotly figure objects originating from custom functions below
    #graphs = [figBarChart,figBarMessagesperCategory]
    graphs = [pieFigureSections,barFigureCompanies,histFigureApplicationDateAfter]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


## define custom functions

def plotypieChart(input_df, col1, pieTitle):
    """
    Create a plotly figure of a Bar chart representing the different patent sections in the dataset.
    input:  a pandas dataframe, a string representing a column name and string as desired title
    output: plotly figure object containing the visualization
    """

    labels = input_df[col1].value_counts().index
    values  = input_df[col1].value_counts().values

    numPatents = input_df.shape[0]


    data = []
    data.append(Pie(labels=labels, values=values))#, title= pieTitle

    layout = Layout(title = pieTitle + " of " + str(numPatents) + " Patents:")

    return Figure(data=data, layout=layout)


def company_countBar(input_df):
    """
    Create a plotly figure of a Bar chart representing the different companies in the dataset.
    Input: dataset df as a pandas dataframe
    Output: plotly figure object containing the visualization
    """
    # extract data for visualization - employ only data with count higher than single digit
    company_count= input_df['company'].value_counts().reset_index(name="count").query("count > 20")
    labels = company_count['index'].values
    values  = company_count['count'].values

    numCompanies =  input_df['company'].value_counts().shape[0]

    data = []
    data.append(Bar(x=labels,y=values,orientation='v'))
    #data.append(Bar(x=company_names,y=company_counts,orientation='h'))

    layout = Layout(title=" There are "+ str(numCompanies) + " companies in the data set. The following have with more than 20 Patents in the Dataset:",
                xaxis=dict(
                    title='',
                    tickangle=20
                ),
                yaxis=dict(
                    title='Patent Count',
                    automargin= True,
                )
            )

    return Figure(data=data, layout=layout)

def date_histogram(input_df,dateCol, histTitle):
    """
    Create a plotly figure of a Bar chart representing the different patent application dates in the dataset
    Input: dataset df as a pandas dataframe
    Output: plotly figure object containing the visualization
    """
    # Convert date columns from object to datetime
    dummy_df = input_df[[dateCol]].apply(pd.to_datetime)

    # number of bins should be an intenger version of the days of the time delta
    binNumber = int( (dummy_df.max()-dummy_df.min()).dt.days[0] )


    data = []
    data.append(Histogram(x = dummy_df[dateCol], nbinsx = binNumber))

    layout = Layout(title=histTitle,
                xaxis=dict(
                    title='Year-Month-Day',
                    tickangle=20
                ),
                yaxis=dict(
                    title='Patent Count',
                    automargin= True,
                )
            )

    return Figure(data=data, layout=layout)


## Execute custom functions to generate ploty figure objects for visualization

pieFigureSections = plotypieChart(df, "section",'CPC Patent Sections in the Analyzed Data Set')
barFigureCompanies = company_countBar(df) #add insert button later for better display?
histFigureApplicationDateAfter = date_histogram(df,'date-appl','Patent Application per Day in the Dataset:')

## Stuff

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()