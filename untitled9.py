from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import plotly.plotly as py
from plotly.figure_factory import create_table
from plotly.graph_objs import Scatter, Figure, Layout
import scipy
import time
from plotly.grid_objs import Grid, Column
import plotly
import json
import csv
import pandas
from nltk.tokenize import word_tokenize
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas_datareader.data as web
from pandas import Series, DataFrame
import seaborn as sns
from datetime import datetime

#start and end dates of time period
start = datetime(2012,11,20)
end = datetime(2017,9,29)
# Tesla ticker
ticker='TSLA'
#we read the data of ticker from yahoo/finance
data=web.DataReader(ticker, 'yahoo', start, end)
#print(data)

#the data can be downloaded from the csv. Since we will somewhat use the dates I 
#used csv to get dates as the column
data=pd.read_csv(r'/Users/alyonarodin/Desktop/python_em_tweets/TSLA.csv')
#percent change calculated using Adj Close (the Close prices after amendments)
percent_change=data['Adj Close'].pct_change()*100
volume=data['Volume']
#we took two columns of the data downloaded from the yahoo website
#print(percent_change[1:])
#print(volume[1:])

#calculated the column(variable) of direction of the return Up and Down
list_up_down=[]
n=0
for value in percent_change:
        #print(value)
        if (value>0.0):
           list_up_down.append('Up')
           n+=1
        else:
           list_up_down.append('Down')
           n+=1

#print(list_up_down) 
#print(data['Date'][1:])  
#created a table : Date, Adj Close, Return, Volume, Direction. Adj Close and
#Return will be highly correlated, but included... 
stock_data=pd.DataFrame(list(zip(data['Date'][1:],data['Adj Close'][1:],percent_change[1:],volume[1:],list_up_down)), columns=['date','AdjClose','percentage_change', 'volumne', 'Direction'])
#print('Maximum percent change: ', format(percent_change.max()*100, '.2f'),'%')
#print(stock_data)

#transfered to .json format to be able to use it in the Tableau
stock_Data_json=stock_data.to_json(orient='table')
outfile = open("/Users/alyonarodin/Desktop/python_em_tweets/stock_Data_json.json", "w")
outfile.write(stock_Data_json)
outfile.close()

#but we plotted it also in Plotly. First is a Adj close prices and second data is return
#define the data
trace_adjClose=go.Scatter(
        x=stock_data['date'],
        y=stock_data['AdjClose'],
        name='Adj Close',
        line=dict(width=1,color='#17BECF'),
        opacity=0.9 
        
        )
trace_perc_change=go.Scatter(
        x=stock_data['date'],
        y=stock_data['percentage_change'],
        name='Return',
        line=dict(width=1,color='#FF5733'),
        opacity=0.9
        )
data=[trace_adjClose, trace_perc_change ]

#plotting 
fig=(data)
py.plot(fig,filename='Adj Close prices of Tesla and Return')


data_elonmusk=pd.read_csv('/users/alyonarodin/Desktop/elonmusk.csv', encoding ='latin1')
#downloaded data of elon musk tweets
#print(data_elonmusk.head())
#dropped few columns
data_elonmusk.drop(labels='row ID', axis=1,inplace=True)
data_elonmusk.drop(labels='Retweet from', axis=1,inplace=True)
data_elonmusk.drop(labels='User', axis=1,inplace=True)
#print(data_elonmusk.head())

#cleaned data-removing signes and making letters lower()
data_elonmusk['Tweet']=data_elonmusk['Tweet'].apply(lambda x:" ".join(x.lower() for x in x.split()))
#print(data_elonmusk['Tweet'].head())
data_elonmusk['Tweet']=data_elonmusk['Tweet'].str.replace('[^\w\s]','')
#print(data_elonmusk['Tweet'].head())

#removed stop words
stop=stopwords.words('english')
data_elonmusk['Tweet']=data_elonmusk['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#print(data_elonmusk['Tweet'].head())

#placed tweets into list
tweets=[tweet for tweet in data_elonmusk['Tweet']]    
#print(tweets[1:4])

#created dictionary and performing sentiment analyses on tweets 
#identifying if the tweet is positive, negative or neutral we appended the dictionary
dict_tweets={}
for sentence in tweets:
    sid = SentimentIntensityAnalyzer()
    ss=sid.polarity_scores(sentence)    
    for k in ss:        
        print('{0}: {1}, '.format(k,ss[k]),end='')
    #print(ss[k])    
    q=ss[k]     
    if q==0.0:
        dict_tweets[sentence]='neu'
    elif q>0.0:
        dict_tweets[sentence]='pos'
    elif q<0.0:
        dict_tweets[sentence]='neg'
#print(dict_tweets)
#print(list(dict_tweets))
#appended the list with the results of pos/neg/neu
list_pos_neg=[]
for value in dict_tweets:
     list_pos_neg.append(dict_tweets[value])
#print(list_pos_neg[1:4])
#create table with Tweets, Time, and pos/neg/neu
DF_list_tweets=pd.DataFrame(list(zip(data_elonmusk['Tweet'], data_elonmusk['Time'],list_pos_neg)), columns=['tweet','time', 'pos/neg/neu'])
#print(DF_list_tweets)
#prepared only the date (without time)
#print(d.head())

list_time=[t.split(' ')[0] for t in data_elonmusk['Time'] ]
#print(list_time[:10])

#transfered to Dataframe the time
DF_time_column=pd.DataFrame(list_time)
#print(DF_time_column)
#calculated the number tweets per each day
new_list_num_tweets=[list_time.count(str(date)) for date in list_time]
     
#print(new_list_num_tweets)
DF_list_tweets=pd.DataFrame(list(zip(data_elonmusk['Tweet'], data_elonmusk['Time'],new_list_num_tweets,list_pos_neg)), columns=['tweet','time','#tweets','pos/neg/neu'])
#print(DF_list_tweets)

#transfered this to json 
number_tweetsPerDay_json=DF_list_tweets.to_json(orient='table')
#print(number_tweetsPerDay_json)
outfile = open("/Users/alyonarodin/Desktop/python_em_tweets/elm_number_tweetsPerDay.json", "w")

outfile.write(number_tweetsPerDay_json)
outfile.close()

#print(DF_list_tweets)
#print(stock_data)
#reversed the data
new_list_pos_neg=list(reversed(list_pos_neg))
new_list_data_tweets=list(reversed(data_elonmusk['Tweet']))
new_list_num_tweets=list(reversed(new_list_num_tweets))
new_list_time=list(reversed(list_time))
#created new table
DF_list_tweets=pd.DataFrame(list(zip(new_list_pos_neg,new_list_data_tweets,new_list_num_tweets,new_list_time)), columns=['ps/neg/neu','tweets','#tweets','time'])
print(DF_list_tweets)

#to count if rt or not (1,0)
rt=[]

for i in DF_list_tweets['tweets']:
    #print(i)
    if 'rt ' in i:
        
        rt.append('1')
    else:
        rt.append('0')
        
print(rt)

#if a tesla in tweet 1 if not 0
tesla_word=[]
for i in DF_list_tweets['tweets']:
    #print(i)
    if 'tesla' in i:
        
        tesla_word.append('1')
    else:
        tesla_word.append('0')
        
print(tesla_word)

#if a spaceX in tweet 1 if not 0
spaceX=[]
for i in DF_list_tweets['tweets']:
    #print(i)
    if 'spacex' in i:
        
        spaceX.append('1')
    else:
        spaceX.append('0')
        
print(spaceX)

DF_list_tweets_new_table=pd.DataFrame(list(zip(new_list_pos_neg,new_list_data_tweets,new_list_num_tweets,new_list_time,rt,tesla_word,spaceX)), columns=['ps/neg/neu','tweets','#tweets','time','rt','tesla','spacex'])
print( DF_list_tweets_new_table)































