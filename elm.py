from plotly.offline import  plot
import plotly.graph_objs as go
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import plotly.plotly as py
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
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
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas_datareader.data as web
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
#start and end dates of time period for our TSLA stock

start = datetime(2012,11,20)
end = datetime(2017,9,29)

# Tesla ticker
ticker='TSLA'

#we read the data of ticker from ...yahoo/finance
data=web.DataReader(ticker, 'yahoo', start, end)

#or

#the data can be downloaded from the .csv file
data=pd.read_csv(r'/Users/alyonarodin/Desktop/python_em_tweets/TSLA.csv')
print(data.head())
#percent change calculated using Adj Close (the Close prices after amendments)
percent_change=data['Adj Close'].pct_change()
volume=data['Volume']

#calculated the column(variable) of direction of the return Up and Down
list_up_down=[]
n=0
for value in percent_change:
        if (value>0.0):
           list_up_down.append('Up')
           n+=1
        else:
           list_up_down.append('Down')
           n+=1
 
#created a table : Date, Adj Close, Return, Volume, Direction 
stock_data=pd.DataFrame(list(zip(data['Date'],data['Adj Close'],percent_change,volume,list_up_down)), columns=['date','AdjClose','percentage_change', 'volumne', 'Direction'])
print(stock_data.head())
print(stock_data.tail())
#print('Maximum percent change: ', format(percent_change.max()*100, '.2f'),'%')

#transfered to .json format to be able to use it in the Tableau to analyse 
stock_Data_json=stock_data.to_json(orient='table')
orig_stock_data=data.to_json(orient='table')
outfile_1 = open("/Users/alyonarodin/Desktop/python_em_tweets/stock_Data_json.json", "w")
outfile_2 = open("/Users/alyonarodin/Desktop/python_em_tweets/orig_stock_data.json", "w")
outfile_1.write(stock_Data_json)
outfile_2.write(stock_Data_json)
outfile_1.close()
outfile_2.close()



#downloaded data of Elon Musk Tweets
data_elonmusk=pd.read_csv('/users/alyonarodin/Desktop/elonmusk.csv', encoding ='latin1')

#drop few columns
data_elonmusk.drop(labels='row ID', axis=1,inplace=True)
data_elonmusk.drop(labels='Retweet from', axis=1,inplace=True)
data_elonmusk.drop(labels='User', axis=1,inplace=True)

#all letters lower case
data_elonmusk['Tweet']=data_elonmusk['Tweet'].apply(lambda x:" ".join(x.lower() for x in x.split()))

#removed the signs
data_elonmusk['Tweet']=data_elonmusk['Tweet'].str.replace('[^\w\s]','')

#removed stop words 
stop=stopwords.words('english')
data_elonmusk['Tweet']=data_elonmusk['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

# we defined each tweet as negative, positive and neutral, using NLP sentiment
#analyzer
tweets=[tweet for tweet in data_elonmusk['Tweet']]    

dict_tweets={}
for sentence in tweets:
    sid = SentimentIntensityAnalyzer()
    ss=sid.polarity_scores(sentence)
    
    
    for k in ss:        
        #print('{0}: {1}, '.format(k,ss[k]),end='')
        q=ss[k]     
        if q==0.0:
            dict_tweets[sentence]='neu'
        elif q>0.0:
            dict_tweets[sentence]='pos'
        elif q<0.0:
            dict_tweets[sentence]='neg'
            
#transfer all values from dictionary to the list to create a table 
list_pos_neg=[]
for value in dict_tweets:
     list_pos_neg.append(dict_tweets[value])
     
#create table with Tweets, Time, and pos/neg/neu
DF_list_tweets=pd.DataFrame(list(zip(data_elonmusk['Tweet'], data_elonmusk['Time'],list_pos_neg)), columns=['tweet','time', 'pos/neg/neu'])
print(DF_list_tweets)
#prepared only the date (without time)
list_date=[t.split(' ')[0] for t in data_elonmusk['Time'] ]
print(list_date)
#calculated the number tweets per each day
new_list_num_tweets=[list_date.count(str(date)) for date in list_date]
print(new_list_num_tweets)

#created table with the column that contain number tweets per day
DF_list_tweets=pd.DataFrame(list(zip(data_elonmusk['Tweet'], list_date,new_list_num_tweets,list_pos_neg)), columns=['tweet','time','#tweets','pos/neg/neu'])
print(DF_list_tweets)
#transfered this to json to use in Tableau
number_tweetsPerDay_json=DF_list_tweets.to_json(orient='table')
outfile = open("/Users/alyonarodin/Desktop/python_em_tweets/elm_number_tweetsPerDay.json", "w")
outfile.write(number_tweetsPerDay_json)
outfile.close()

#reversed the data (be its stock table or tweets table)
new_list_pos_neg=list(reversed(DF_list_tweets['pos/neg/neu']))
new_list_tweets=list(reversed(DF_list_tweets['tweet']))
new_list_num_tweets=list(reversed(DF_list_tweets['#tweets']))
new_list_date=list(reversed(DF_list_tweets['time']))
print(new_list_pos_neg)
print(new_list_tweets)
print(new_list_num_tweets)
print(new_list_date)

#created new table with reversed data
DF_list_tweets=pd.DataFrame(list(zip(new_list_pos_neg,new_list_tweets,new_list_num_tweets,new_list_date)), columns=['pos/neg/neu','tweets','#tweets','time'])
print(DF_list_tweets)

#to see if the words are in tweets or not if rt or not (1,0)
rt=['1' if 'rt' in i else '0' for i in DF_list_tweets['tweets']]

#if a tesla in tweet 1 if not 0
tesla_word=['1' if 'tesla' in i else '0' for i in DF_list_tweets['tweets']]

#if a spaceX in tweet 1 if not 0
spaceX=['1' if 'spacex' in i else '0' for i in DF_list_tweets['tweets']]

#if a model in tweet 1 if not 0
model=['1' if 'model' in i else '0' for i in DF_list_tweets['tweets']]

#added the table
DF_list_tweets_new_table=pd.DataFrame(list(zip(new_list_pos_neg, new_list_num_tweets,new_list_date,rt,tesla_word,spaceX,model)), columns=['pos/neg/neu','#tweets','time','rt','tesla','spacex','model'])
print(DF_list_tweets_new_table)
print(stock_data)

tw=list(DF_list_tweets_new_table['time'])
s=list(stock_data['date'])
print(tw)
print(s)

#created 4 lists to append 
list_adjCl=[]
list_return=[]
list_volume=[]
list_direction=[]
print()
#appended list with new columns, based on the date - to match
for i in tw:
    if i in s:
        ind=s.index(i)
        list_adjCl.append(stock_data['AdjClose'][ind])
        list_return.append(abs(stock_data['percentage_change'][ind]))
        list_volume.append(stock_data['volumne'][ind])
        list_direction.append(stock_data['Direction'][ind])
    else:
        list_adjCl.append('NaN')    
        list_return.append('NaN') 
        list_volume.append('NaN') 
        list_direction.append('NaN') 
print(list_adjCl)
print(list_return)
print(list_volume)
print(list_direction)
#created new table
DF_list_tweets_new_table=pd.DataFrame(list(zip(new_list_pos_neg,new_list_num_tweets,new_list_date,rt,tesla_word,spaceX,model,list_adjCl,list_return,list_volume,list_direction)), columns=['pos/neg/neu','#tweets','time','rt','tesla','spacex','model','list_adjCl','list_return','list_volume','list_direction'])
print(DF_list_tweets_new_table)
#since we have values NAN, we delete them. As there are few tweets per day 
#se assigned the values from table stock_data to those tweets if the dates match. 
d=DF_list_tweets_new_table[DF_list_tweets_new_table.list_adjCl!= 'NaN']
print(d)

#transfered this to json 
table_tweets=d.to_json(orient='table')
outfile = open("/Users/alyonarodin/Desktop/python_em_tweets/table_tweets.json", "w")
outfile.write(table_tweets)
outfile.close()

#looking at the data we can notice that we have categorical and numerical data
#actually we will look for direction which is also categorical and binary.
#for that reason for our model we will use logit or SVM, etc as 
#its for classificaiton (supervised algorithm)

#we remove the dates
new_table=d.drop(['time'],axis=1)
print(new_table)

#since we predict direction UP/DOWN we need to convert it
#but first we plot
new_table['list_direction'].value_counts()
sns.countplot(x='list_direction', data=new_table, palette="coolwarm")
plt.show()
#we have approximately the same amount of ups and downs

#converted Direction values 'Up' and 'Down' to 0 and 1
N_Direction = LabelEncoder()
new_table['list_direction_code'] = N_Direction.fit_transform(new_table['list_direction'])
new_table[["list_direction", "list_direction_code"]].head()
new_table=new_table.drop(['list_direction'],axis=1)
print(new_table.head())

#next we look at the pos/neg/neu
#its a categorical data (looks like nominal) and we can seperate it into dummy 
#variables as well we create dummy variables for columns rt, tesla, spacex,
#model
dummy_Pos_Neg=pd.get_dummies(new_table['pos/neg/neu'],prefix='pos/neg/neu')
dummy_rt=pd.get_dummies(new_table['rt'],prefix='rt')
dummy_tesla=pd.get_dummies(new_table['tesla'], prefix='tesla')
dummy_spacex=pd.get_dummies(new_table['spacex'],prefix='spacex')
dummy_model=pd.get_dummies(new_table['model'],prefix='model')
new_data_w_dummy=pd.concat([dummy_Pos_Neg,dummy_rt,dummy_tesla,dummy_spacex,dummy_model,new_table['list_adjCl'],new_table['list_return'],new_table['list_volume'],new_table['list_direction_code']], axis=1)
print(new_data_w_dummy)

#as we created dummy variables, we also have now multicolinearity. 
#we need to drop one column in each dummy category for that reason
#lets drop them
new_data_w_dummy_=new_data_w_dummy.drop(['pos/neg/neu_neg'],axis=1)
new_data_w_dummy_=new_data_w_dummy_.drop(['rt_0'],axis=1)
new_data_w_dummy_=new_data_w_dummy_.drop(['tesla_0'],axis=1)
new_data_w_dummy_=new_data_w_dummy_.drop(['spacex_0'],axis=1)
new_data_w_dummy_=new_data_w_dummy_.drop(['model_0'],axis=1)
print(new_data_w_dummy_)

new_table_with_dummy_=new_data_w_dummy_.dropna()
print(new_table_with_dummy_)


#there are couple columns in the table which we need to normalize 
def norm(data):
    new_list=[(i-min(list(data)))/(max(list(data))-min(list(data))) for i in list(data)]    
    return(new_list)

def absol(data)    :
    new_list=[abs(i) for i in list(data) ]
    return(new_list)

if __name__ == '__main__':  
    new_data_w_dummy_adjCl = norm(new_data_w_dummy_['list_adjCl'])
    new_data_w_dummy_volume = norm(new_data_w_dummy_['list_volume'])
    
    print(new_data_w_dummy_adjCl[1:4])
    print(new_data_w_dummy_volume[1:4])
    
    
    
new_data_w_dummy_ =new_data_w_dummy_.drop(['list_adjCl'],axis=1)   
new_data_w_dummy_ =new_data_w_dummy_.drop(['list_volume'],axis=1) 
list_with_norm_data=pd.DataFrame(list(zip(new_data_w_dummy_adjCl,new_data_w_dummy_volume,new_data_w_dummy_['list_return'],new_data_w_dummy_['pos/neg/neu_neu'],new_data_w_dummy_['pos/neg/neu_pos'],new_data_w_dummy_['rt_1'],new_data_w_dummy_['tesla_1'],new_data_w_dummy_['spacex_1'],new_data_w_dummy_['model_1'],new_table['list_direction_code'])), columns=['list_adjCl','list_volume','list_return','pos/neg/neu_neu','pos/neg/neu_pos','rt','tesla','spacex','model','list_direction'])
print(list_with_norm_data)
#sns.heatmap(list_with_norm_data.corr())
#corr =list_with_norm_data.corr()
#print(corr)    

#preparing data we dropped the rows that have NAN values but
#matching the dates but it has its disadvantages:
#1.we can end up removing too much data - actually this is what we had 
#but we removed a lot of data from the table that had tweets
#since Elon Musk tweeted sometimes 10 times per day.
#2. we can lose valuable information-true for us too
#because we might removed the tweets that effected the direction
#if he tweeted over the weekend
#but we did not removed any values from our stock data
#we will lieave as we have and look what we get.
#however there is a different approch to deal with missing data
#-interpolation technique - imputation-replace the missing value by the mean 
#value of the entire feature column 
#for the future we can make it and to see the difference
#(in the second part of report)
# for the categorical data we can use - most_frequent

X,y=new_table_with_dummy_.iloc[:,:9].values,new_table_with_dummy_.iloc[:,9].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

stdsc = StandardScaler()
stdsc.fit(X_train)
X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)


LogisticRegression(penalty='l1')
lr = LogisticRegression(penalty='l1', C=0.5)    
lr.fit(X_train_std, y_train) 
print('Training accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test)) 
print(lr.intercept_    )
print(lr.coef_)
    
fig = plt.figure()   
ax = plt.subplot(111)    
colors = ['blue', 'green', 'red', 'cyan','magenta', 'yellow', 'black','pink', 'lightgreen']
weights, params = [], []    
for c in np.arange(-4, 6,  dtype=float):    
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[0])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=new_table_with_dummy_.columns[column+1], color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=2)    
plt.xlim([10**(-3), 10**5])    
plt.ylabel('weight coefficient')   
plt.xlabel('C')    
plt.xscale('log')    
plt.legend(loc='upper left')    
ax.legend(loc='upper center',    
    bbox_to_anchor=(1.38, 1.03),
    ncol=1, fancybox=True)
plt.show()    
    
#(another way to select features is SBS)
#assesing feature importance by using random forests

#(another way to select features is SBS)
#assesing feature importance by using random forests

feat_labels = new_table_with_dummy_.columns[0:9]    
forest = RandomForestClassifier(n_estimators=10000,random_state=0,n_jobs=-1)
forest.fit(X_train, y_train)    
importances = forest.feature_importances_    
indices = np.argsort(importances)[::-1]    
for f in range(X_train.shape[1]):    
    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[f],importances[indices[f]]))
plt.title('Feature Importances')    
plt.bar(range(X_train.shape[1]),importances[indices], color='lightblue', align='center')  
plt.xticks(range(X_train.shape[1]),feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])    
plt.tight_layout()   
plt.show() 








