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
from nltk.tokenize import word_tokenize
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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
#downloaded data
data_elonmusk=pd.read_csv('/users/alyonarodin/Desktop/elonmusk.csv', encoding ='latin1')
#print(data_elonmusk.head())

#drop few columns
data_elonmusk.drop(labels='row ID', axis=1,inplace=True)
data_elonmusk.drop(labels='Retweet from', axis=1,inplace=True)
data_elonmusk.drop(labels='User', axis=1,inplace=True)
#print(data_elonmusk.head())

#all letters lower case
data_elonmusk['Tweet']=data_elonmusk['Tweet'].apply(lambda x:" ".join(x.lower() for x in x.split()))
print(data_elonmusk['Tweet'].head())

#removed the signs
data_elonmusk['Tweet']=data_elonmusk['Tweet'].str.replace('[^\w\s]','')
print(data_elonmusk['Tweet'].head())

#removed stop words 
stop=stopwords.words('english')
data_elonmusk['Tweet']=data_elonmusk['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(data_elonmusk['Tweet'].head())

# we defined each tweet as negative, positive and neutral
tweets=[tweet for tweet in data_elonmusk['Tweet']]    
print(tweets[1:4])

# defined each tweet as positive, negative or neutral
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
rt=['1' if 'rt' in i else '0' for i in DF_list_tweets['tweets']]
#print(rt)

#if a tesla in tweet 1 if not 0
tesla_word=['1' if 'tesla' in i else '0' for i in DF_list_tweets['tweets']]
#print(tesla_word)
#if a spaceX in tweet 1 if not 0
spaceX=['1' if 'spacex' in i else '0' for i in DF_list_tweets['tweets']]
#print(spaceX)
#if a model in tweet 1 if not 0
model=['1' if 'model' in i else '0' for i in DF_list_tweets['tweets']]
#print(model)

DF_list_tweets_new_table=pd.DataFrame(list(zip(new_list_pos_neg,new_list_data_tweets,new_list_num_tweets,new_list_time,rt,tesla_word,spaceX,model)), columns=['ps/neg/neu','tweets','#tweets','time','rt','tesla','spacex','model'])
print( DF_list_tweets_new_table)


#the most common words
freq = pd.Series(' '.join(data_elonmusk['Tweet']).split()).value_counts()[:40]
print(freq)
print(list(freq.index))
print(list(freq))

#transfered the most used words into .json and saved it to open in Tableau
freq_json=freq.to_json(orient='table')
print(freq_json)
outfile = open("/Users/alyonarodin/Desktop/plotly_em/twitter_elm_json.json", "w")

outfile.write(freq_json)
outfile.close()    

#calcualated the different of using the most frequent words and other words of the 4 most frequent words
number=freq[0]
list_diff=[number-x for x in freq ]
print(list_diff)

#created the table of the words, frequency of each word and differences 
DF_list_words=pd.DataFrame(list(zip(freq.index, freq, list_diff)), columns=['word', 'quantity','difference'])
print(DF_list_words)


#created table in Plotly
table=create_table(DF_list_words)
plot(table)

#plotting-assigned the columns to each variable to plot, colors, names 
data0 = go.Bar(x=DF_list_words.word,
            y=DF_list_words.quantity, 
            name='Frequency',
            marker=dict(color='#829db4'))
data1 = go.Bar(x=DF_list_words.word,
            y=DF_list_words.difference,
            name='Difference',
            marker=dict(color='#f1f1f1'))

#combined two variables
data=[data0,data1]
#layout bar plot
layout=go.Layout(
    barmode='stack')
fig = go.Figure(data=data, layout=layout)   
#plotted  with Plotly
plot(fig, filename='word frequency')


#splitted data and time
d=data_elonmusk['Time']
print(d.head())

list_time=[t.split(' ')[0] for t in d ]
print(list_time[:10])

#converted list of dates into data frame
DF_time_column=pd.DataFrame(list_time)
print(DF_time_column)

#splitted month, date, year to get year
list_year=[ '20'+str(ch).split('/')[2] for ch in DF_time_column[0]]
#print(list_year)

DF_list_words=pd.DataFrame(list(zip(data_elonmusk['Tweet'], list_year)), columns=['Tweet', 'year'])
print(DF_list_words)

#calculate number tweets per year

list_number_tweets=[]
total_2017=0
total_2016=0
total_2015=0
total_2014=0
total_2013=0
total_2012=0

for i in DF_list_words['year']:
    if i=='2017':
        total_2017+=1
    elif i=='2016':
        total_2016+=1   
    elif i=='2015':
        total_2015+=1
    elif i=='2014':
        total_2014+=1
    elif i=='2013':
        total_2013+=1
    elif i=='2012':
        total_2012+=1
    else:
        False

#append the list with number tweets
for j in DF_list_words['year']:
    if j=='2017':
        list_number_tweets.append(total_2017)
    elif j=='2016':
        list_number_tweets.append(total_2016)  
    elif j=='2015':
        list_number_tweets.append(total_2015)
    elif j=='2014':
        list_number_tweets.append(total_2014)
    elif j=='2013':
        list_number_tweets.append(total_2013)
    elif j=='2012':
        list_number_tweets.append(total_2012)
    else:
        False
print(list_number_tweets)

#created new table
DF_list_words=pd.DataFrame(list(zip(DF_list_words['Tweet'],DF_list_words['year'],list_number_tweets)), columns=['Tweet', 'year','number_tweets'])
print(DF_list_words)



#the most frequent words in each year
freq_2017 = pd.Series(' '.join(DF_list_words['Tweet'][DF_list_words['year']=='2017']).split()).value_counts()[:20]
freq_2016 = pd.Series(' '.join(DF_list_words['Tweet'][DF_list_words['year']=='2016']).split()).value_counts()[:20]
freq_2015 = pd.Series(' '.join(DF_list_words['Tweet'][DF_list_words['year']=='2015']).split()).value_counts()[:20]
freq_2014 = pd.Series(' '.join(DF_list_words['Tweet'][DF_list_words['year']=='2014']).split()).value_counts()[:20]
freq_2013 = pd.Series(' '.join(DF_list_words['Tweet'][DF_list_words['year']=='2013']).split()).value_counts()[:20]
freq_2012 = pd.Series(' '.join(DF_list_words['Tweet'][DF_list_words['year']=='2012']).split()).value_counts()[:20]

#created the years for each case
list_2017=pd.DataFrame(['2017']*20)
list_2016=pd.DataFrame(['2016']*20)
list_2015=pd.DataFrame(['2015']*20)
list_2014=pd.DataFrame(['2014']*20)
list_2013=pd.DataFrame(['2013']*20)
list_2012=pd.DataFrame(['2012']*20)
#concatenated the years and data
new_years=pd.concat([list_2017,list_2016,list_2015,list_2014,list_2013,list_2012])
new_data_set_20_words=pd.concat([freq_2017,freq_2016,freq_2015,freq_2014,freq_2013,freq_2012])
print(new_data_set_20_words)
#put everything into one table
DF_list_words_new=pd.DataFrame(list(zip(new_years[0],new_data_set_20_words,new_data_set_20_words.index)), columns=['Year', 'Words','Number'])
print(DF_list_words_new)

table1=create_table(DF_list_words_new)
plot(table1)
twitter_data=DF_list_words_new.to_json(orient='table')

outfile=open("/Users/alyonarodin/Desktop/plotly_em/twitter_data.json", "w")
outfile.write(twitter_data)
outfile.close()







#second Plot made in plotly

spacex=[]
count=0
for sx in freq_2017.index:               
        if sx=='spacex':              
            spacex.append(freq_2017[count])
            count+=1
        else:
            count+=1
count=0  
for sx in freq_2016.index:               
        if sx=='spacex':              
            spacex.append(freq_2016[count])
            count+=1
        else:
            count+=1
count=0  
for sx in freq_2015.index:               
        if sx=='spacex':              
            spacex.append(freq_2015[count])
            count+=1
        else:
            count+=1
count=0       
for sx in freq_2014.index:               
        if sx=='spacex':              
            spacex.append(freq_2014[count])
            count+=1
        else:
            count+=1

count=0           
for sx in freq_2013.index:               
        if sx=='spacex':              
            spacex.append(freq_2013[count])
            count+=1
        else:
            count+=1
count=0          
for sx in freq_2012.index:               
        if sx=='spacex':              
            spacex.append(freq_2012[count])
            count+=1
        else:
            count+=1
  
print(spacex)


tesla=[]
count=0
for tl in freq_2017.index:               
        if tl=='tesla':              
            tesla.append(freq_2017[count])
            count+=1
        else:
            count+=1
count=0           
for tl in freq_2015.index:               
        if tl=='tesla':              
            tesla.append(freq_2015[count])
            count+=1
        else:
            count+=1
count=0           
for tl in freq_2014.index:               
        if tl=='tesla':              
            tesla.append(freq_2014[count])
            count+=1
        else:
            count+=1
count=0           
for tl in freq_2013.index:               
        if tl=='tesla':              
            tesla.append(freq_2013[count])
            count+=1
        else:
            count+=1
count=0           
for tl in freq_2012.index:               
        if tl=='tesla':              
            tesla.append(freq_2012[count])
            count+=1
        else:
            count+=1 

print(tesla)

rt=[]
count=0
for r in freq_2017.index:               
        if r=='rt':              
            rt.append(freq_2017[count])
            count+=1
        else:
            count+=1
count=0
for r in freq_2016.index:               
        if r=='rt':              
            rt.append(freq_2016[count])
            count+=1
        else:
            count+=1
count=0           
for r in freq_2015.index:               
        if r=='rt':              
            rt.append(freq_2015[count])
            count+=1
        else:
            count+=1
count=0           
for r in freq_2014.index:               
        if r=='rt':              
            rt.append(freq_2014[count])
            count+=1
        else:
            count+=1
count=0           
for r in freq_2013.index:               
        if r=='rt':              
            rt.append(freq_2013[count])
            count+=1
        else:
            count+=1
count=0           
for r in freq_2012.index:               
        if r=='rt':              
            rt.append(freq_2012[count])
            count+=1
        else:
            count+=1 

print(rt)


year=['2017','2016','2015','2014','2013','2012']

DF_list_words_tesla_spacex=pd.DataFrame(list(zip(year,rt, spacex, tesla)), columns=['Year','RT','Spacex','Tesla'])
print(DF_list_words_tesla_spacex)
colorscale=[[0,'#390b00'],[.5, '#ffffff'],[1, '#ffffff']]
table1=create_table(DF_list_words_tesla_spacex, colorscale=colorscale )
table1.layout.width=250
#plot(table1)
N=150
x=DF_list_words_tesla_spacex['Year']


trace1 = go.Scatter(
    x = DF_list_words_tesla_spacex['Year'],
    y = DF_list_words_tesla_spacex['RT'],
    mode = 'lines',
    name = 'RT',
    line=dict(width=2, color='#acb0b8')
)
trace2 = go.Scatter(
    x = DF_list_words_tesla_spacex['Year'],
    y = DF_list_words_tesla_spacex['Spacex'],
    mode = 'lines+markers',
    name = 'spacex',
    line=dict(width=1, color='#444444')
)
trace3 = go.Scatter(
    x = DF_list_words_tesla_spacex['Year'],
    y = DF_list_words_tesla_spacex['Tesla'],
    mode = 'lines+markers',
    name = 'tesla',
    line=dict(width=1, color='#17BECF')
)
data = [trace1, trace2, trace3]
py.iplot(data, filename='')

tw=list(DF_list_tweets_new_table['time'])
s=list(stock_data['date'])
#created 4 lists to append 
list_adjCl=[]
list_return=[]
list_volume=[]
list_direction=[]
#appended list with new columns, based on the date- to match
for i in tw:
    if i in s:
        ind=s.index(i)
        list_adjCl.append(stock_data['AdjClose'][ind])
        list_return.append(stock_data['percentage_change'][ind])
        list_volume.append(stock_data['volumne'][ind])
        list_direction.append(stock_data['Direction'][ind])
    else:
        list_adjCl.append('NaN')    
        list_return.append('NaN') 
        list_volume.append('NaN') 
        list_direction.append('NaN') 
#created new table
DF_list_tweets_new_table=pd.DataFrame(list(zip(new_list_pos_neg,new_list_time,rt,tesla_word,spaceX,model,list_adjCl,list_return,list_volume,list_direction)), columns=['ps/neg/neu','time','rt','tesla','spacex','model','list_adjCl','list_return','list_volume','list_direction'])
print(DF_list_tweets_new_table)
#since we have values NAN, we delete them. As there are few tweets per day 
#se assigned the values from table stock_data to those tweets if the dates match. 
d=DF_list_tweets_new_table[DF_list_tweets_new_table.list_adjCl!= 'NaN']
#print(len(d))

#next we perform analyses for the table just has been created















