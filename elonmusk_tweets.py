from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from plotly.graph_objs import Scatter, Figure, Layout
from plotly.tools import FigureFactory as FF
import scipy
import plotly.plotly as py
from plotly.grid_objs import Grid, Column
import time
import plotly.plotly as py
from plotly.grid_objs import Grid, Column
from plotly.tools import FigureFactory as FF 
import plotly
import pandas as pd
import time
from wordcloud import WordCloud

data_elonmusk=pd.read_csv('/users/alyonarodin/Desktop/elonmusk.csv', encoding ='latin1')
print(data_elonmusk.head())
data_elonmusk.drop(labels='row ID', axis=1,inplace=True)
data_elonmusk.drop(labels='Retweet from', axis=1,inplace=True)
data_elonmusk.drop(labels='User', axis=1,inplace=True)
print(data_elonmusk.head())
data_elonmusk['Tweet']=data_elonmusk['Tweet'].apply(lambda x:" ".join(x.lower() for x in x.split()))
print(data_elonmusk['Tweet'].head())
data_elonmusk['Tweet']=data_elonmusk['Tweet'].str.replace('[^\w\s]','')
print(data_elonmusk['Tweet'].head())
stop=stopwords.words('english')
data_elonmusk['Tweet']=data_elonmusk['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(data_elonmusk['Tweet'].head())
freq = pd.Series(' '.join(data_elonmusk['Tweet']).split()).value_counts()[:40]
print(freq)
print(list(freq.index))
print(list(freq))

plt.axis('off')
plt.savefig('graph.png')
freq_=pd.Series(' '.join(data_elonmusk['Tweet']).split()).value_counts()[-5871:]
#print(freq_)
freq_=list(freq_.index)
data_elonmusk['Tweet'].apply(lambda x: ' '.join(x for x in x.split() if x not in freq_))
print(data_elonmusk['Tweet'].head())

list_diff=[]
number=freq[0]
#print(number)
for x in freq:
    #print(x)
    new_number=number-x
    list_diff.append(new_number)
print(list_diff)
DF_list_words=pd.DataFrame(list(zip(freq.index, freq,list_diff)), columns=['word', 'quantity','difference'])
print(DF_list_words)

table=FF.create_table(DF_list_words)
#plot(table, filename='word frequency')

data0 = go.Bar(x=DF_list_words.word,
            y=DF_list_words.quantity, 
            name='Frequency',
            marker=dict(color='#191970'))
data1 = go.Bar(x=DF_list_words.word,
            y=DF_list_words.difference,
            name='Difference',
            marker=dict(color='#D3D3D3'))
data=[data0,data1]
layout=go.Layout(
    barmode='stack')
fig = go.Figure(data=data, layout=layout)    
plot(fig, filename='word frequency')

print(data_elonmusk.head())

DF_list_words_table=pd.DataFrame(list(zip(words,year,tweets,num_words)), columns=['words', 'year','tweets','num_words'])
print(DF_list_words_table.head())

plt.show()

d=data_elonmusk['Time']
print(d.head())
list_time=[]
for time in d:
    y=time.split(' ')
    list_time.append(y[0])
print(list_time[:10])


DF_time_column=pd.DataFrame(list_time)
print(DF_time_column)
list_year=[]


for ch in DF_time_column[0]:
    c=str(ch).split('/')
    list_year.append('20'+c[2])
print(list_year)

DF_list_words=pd.DataFrame(list(zip(data_elonmusk['Tweet'], list_year)), columns=['Tweet', 'year'])
print(DF_list_words)

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
    if i=='2016':
        total_2016+=1   
    if i=='2015':
        total_2015+=1
    if i=='2014':
        total_2014+=1
    if i=='2013':
        total_2013+=1
    if i=='2012':
        total_2012+=1

for j in DF_list_words['year']:
    if j=='2017':
        list_number_tweets.append(total_2017)
    if j=='2016':
        list_number_tweets.append(total_2016)  
    if j=='2015':
        list_number_tweets.append(total_2015)
    if j=='2014':
        list_number_tweets.append(total_2014)
    if j=='2013':
        list_number_tweets.append(total_2013)
    if j=='2012':
        list_number_tweets.append(total_2012)
print(list_number_tweets)


DF_list_words=pd.DataFrame(list(zip(DF_list_words['Tweet'],DF_list_words['year'],list_number_tweets)), columns=['Tweet', 'year','number_tweets'])
print(DF_list_words)

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

words=['rt','rt','rt','rt','rt','rt','tesla','tesla','tesla','tesla','tesla','tesla','spacex','spacex','spacex','spacex','spacex','spacex']
year=['2012','2013','2014','2015','2016','2017','2012','2013','2014','2015','2016','2017','2012','2013','2014','2015','2016','2017']
tweets=['55','479','231','436','934','1083','55','479','231','436','934','1083','55','479','231','436','934','1083']
num_words=[7,59,'44','108','181','130','7','78','31','44','107','78','5','25','17','56','63','87']




