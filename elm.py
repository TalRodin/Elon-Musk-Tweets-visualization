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
from wordcloud import WordCloud
import json
import csv


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
list_diff=[]
number=freq[0]
for x in freq:
    new_number=number-x
    list_diff.append(new_number)
print(list_diff)

#created the table of the words, frequency of each word and differences 
DF_list_words=pd.DataFrame(list(zip(freq.index, freq, list_diff)), columns=['word', 'quantity','difference'])
print(DF_list_words)
#created table in plotly
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
#plotted  
plot(fig, filename='word frequency')

#splitted data and time
d=data_elonmusk['Time']
print(d.head())
list_time=[]
for t in d:
    y=t.split(' ')
    list_time.append(y[0])
print(list_time[:10])
#converted list of dates into data frame
DF_time_column=pd.DataFrame(list_time)
print(DF_time_column)

#splitted month, date, year to get year
list_year=[]
for ch in DF_time_column[0]:
    c=str(ch).split('/')
    list_year.append('20'+c[2])
print(list_year)

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
print(freq_2017)
freq_2016 = pd.Series(' '.join(DF_list_words['Tweet'][DF_list_words['year']=='2016']).split()).value_counts()[:20]
print(freq_2016)
freq_2015 = pd.Series(' '.join(DF_list_words['Tweet'][DF_list_words['year']=='2015']).split()).value_counts()[:20]
print(freq_2015)
freq_2014 = pd.Series(' '.join(DF_list_words['Tweet'][DF_list_words['year']=='2014']).split()).value_counts()[:20]
print(freq_2014)
freq_2013 = pd.Series(' '.join(DF_list_words['Tweet'][DF_list_words['year']=='2013']).split()).value_counts()[:20]
print(freq_2013)
freq_2012 = pd.Series(' '.join(DF_list_words['Tweet'][DF_list_words['year']=='2012']).split()).value_counts()[:20]
print(freq_2012)

#created the years for each case
list_2017=pd.DataFrame(['2017']*20)
print(list_2017)
list_2016=pd.DataFrame(['2016']*20)
print(list_2016)
list_2015=pd.DataFrame(['2015']*20)
print(list_2015)
list_2014=pd.DataFrame(['2014']*20)
print(list_2014)
list_2013=pd.DataFrame(['2013']*20)
print(list_2013)
list_2012=pd.DataFrame(['2012']*20)
print(list_2012)
#concatenated the years and data
new_years=pd.concat([list_2017,list_2016,list_2015,list_2014,list_2013,list_2012])
new_data_set_20_words=pd.concat([freq_2017,freq_2016,freq_2015,freq_2014,freq_2013,freq_2012])
print(new_data_set_20_words)
#put everything into one table
DF_list_words_new=pd.DataFrame(list(zip(new_years[0],new_data_set_20_words,new_data_set_20_words.index)), columns=['Year', 'words','Number'])
print(DF_list_words_new)

table1=create_table(DF_list_words_new)
plot(table1)
twitter_data=DF_list_words_new.to_json(orient='table')

outfile=open("/Users/alyonarodin/Desktop/plotly_em/twitter_data.json", "w")
outfile.write(twitter_data)
outfile.close()

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
for tl in freq_2016.index:               
        if tl=='tesla':              
            tesla.append(freq_2016[count])
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

