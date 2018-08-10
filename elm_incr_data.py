#changing NaN values 
from sklearn.preprocessing import Imputer 
from array import *
import statistics
import pandas as pd

DF_list_tweets_new_table

list_d=[]
for i in DF_list_tweets_new_table['list_direction']:
    if i =='Up':
        list_d.append(1)
    elif i =='Down':
        list_d.append(0)
    elif i=='NaN':    
        list_d.append('NaN')
#print(list_d)


count_up=0
count_down=0

for i in list_d:
    if i==1:
        count_up+=1
    elif i==0:
        count_down+=1
#print(count_up)
#print(count_down)

list_U_D=[]
 
if count_up>count_down:
    for i in list_d:
        if i =='NaN':
            list_U_D.append(1)
        else:
            list_U_D.append(i)
#print(list_U_D) 


list_adjCl_mean=mean_(DF_list_tweets_new_table['list_adjCl'])
list_volume_mean=mean_(DF_list_tweets_new_table['list_volume'])
print(list_return_mean)


def append_(x,m):
    list_=[]
    for i in x:
        if i =='NaN':
            list_.append(m)
        else:
            list_.append(i)
    return (list_)

list_1=append_(DF_list_tweets_new_table['list_adjCl'],list_adjCl_mean)
list_3=append_(DF_list_tweets_new_table['list_volume'],list_volume_mean)

#print(list_1)
#print(list_3)

DF_list_tweets_new_table=pd.DataFrame(list(zip(DF_list_tweets_new_table['pos/neg/neu'],DF_list_tweets_new_table['#tweets'],DF_list_tweets_new_table['time'],DF_list_tweets_new_table['rt'],DF_list_tweets_new_table['tesla'],DF_list_tweets_new_table['spacex'],DF_list_tweets_new_table['model'],list_1,list_3,list_U_D)), columns=['pos/neg/neu','#tweets','time','rt','tesla','spacex','model','list_adjCl','list_volume','list_direction'])
print(DF_list_tweets_new_table)


count_up=0
count_down=0

for i in DF_list_tweets_new_table['list_direction']:
    if i==1:
        count_up+=1
    elif i==0:
        count_down+=1
print(count_up)
print(count_down)   

#ups more then downs --> we need increase data of downs


    
