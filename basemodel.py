#imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import tensorflow as tf
from tensorflow import keras
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy ='mean')
from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler()
from sklearn.model_selection import train_test_split



pd.options.display.max_columns = 100 
pd.options.display.max_rows = 300 

fileName = 'PLL Stats Master - Player Game Logs.csv'
playerdata = pd.read_csv(fileName)
playerdata = playerdata[0:3770]

fileName2 = 'PLL Stats Master - Team Game Logs.csv'
teamdata = pd.read_csv(fileName2)
teamdata = teamdata[teamdata['Season'] != 2022]

#data cleaning

teamdata['Avg Shot Dist'] = teamdata['Average Shot Distance'].replace('#DIV/0!' , 10.265)
teamdata['Avg Shot Dist'] = teamdata['Avg Shot Dist'].astype(float)

team = teamdata[['Season','Week','Game','Team','Opponent', 
                 'Shots','Goals','Efficiency', 
                 'Possession %',
                 'Settled Goal','FB Goals','Assisted Goals','Shot Quality Ratio', 
                 'D Efficiency','Turnovers', 'Score Against',
                 'Expected Goals','Avg Shot Dist','Margin', 'Settled Goals agaisnt',
                 'Score','Result', 'Save %']]
team['Shot%'] = team['Goals'] / team['Shots']
team['W/L'] = pd.get_dummies(team.loc[:,'Result'])[['W']]


x = team[['Shots','Shot%','Efficiency','Possession %',
                 'Settled Goal','FB Goals','Assisted Goals','Shot Quality Ratio', 
                 'D Efficiency','Turnovers', 'Score Against',
                 'Expected Goals','Avg Shot Dist','Margin', 'Save %' , 'Settled Goals agaisnt']]

x = imp.fit_transform(x)


ylinear = np.array(team[['Score']]).ravel()
x1 = stdscaler.fit_transform(x)
x_train, x_test, y_train , y_test = train_test_split(x1,ylinear, test_size = .2, random_state = 42)

x_valid, x_train1 = x_train[:15] , x_train[15:]
y_valid, y_train1 = y_train[:15] , y_train[15:]

#modeling

model = keras.models.Sequential([keras.layers.Dense(50 , activation = 'relu' , input_shape = x_train1.shape[1:]),
                                keras.layers.Dense(100 , activation = 'relu'),
                                keras.layers.Dense(100 , activation = 'relu'),
                                keras.layers.Dense(125 , activation = 'relu'),
                                keras.layers.Dense(100 , activation = 'relu'),
                                keras.layers.Dense(1)])

model.compile(loss = 'mean_squared_error' , 
             optimizer = keras.optimizers.SGD(learning_rate = .01, clipnorm = 1) ,)

history = model.fit(x_train1, y_train1, epochs = 500,
                  validation_data = (x_valid , y_valid))

#scoring

mse_training = model.evaluate(x_train1 , y_train1)
y_pred = model.predict(x_train1)

y_test = y_test.reshape(40,1)
test_mse = model.evaluate(x_test, y_test)
print (test_mse)


#plotting and replacement functions




def plotter(pred , true):

    plt.plot(true , c = 'green', label = 'True')
    plt.plot(pred, c = 'yellow', label = 'Predicted', linestyle = "--")
    plt.ylim([0,30])
    plt.xlabel('Games')
    plt.ylabel('Goals')
    plt.legend()
    plt.xticks(ticks = [],labels = [])
    plt.savefig("redwoodsplot.jpg")
    plt.show()



def generate (count, mini , maxi, average):
    arr = []
    diff = 1
    while len(arr) < count-1:
        if mini <= average - diff and average + diff <= maxi:
            arr.append(average - diff)
            arr.append(average + diff)
            diff += 1
        else:
            arr.append(average)
            diff = 1
    if len(arr) < count:
        arr.append(average)
    return arr


def replace(data , metric , avg):
    data = data
    length = len(data)
    league = data.describe()
    
    mi1 = league[[metric]][3:4]
    mi = mi1[metric][0] 
    
    ma1 = league[[metric]][7:8] 
    ma = ma1[metric][0]
    
    avg = avg
    
    data.drop(columns = [metric])
    
    new = generate(length , mi , ma , avg )
    
    data[metric] = new
    
    return data
    

def project(initial, metric1 , avg1 , metric2, avg2, metric3 , avg3):
    
    t1 = replace(initial , metric1 , avg1)
    t2 = replace(t1 , metric2 , avg2)
    t3 = replace(t2, metric3 , avg3)
    
    t3 = t3.drop(columns = ['Season' , 'Week' , 'Game' , 'Team' , 'Opponent'])
    t3 = t3[['Shots','Shot%','Efficiency','Possession %',
          'Settled Goal','FB Goals','Assisted Goals','Shot Quality Ratio', 
          'D Efficiency','Turnovers', 'Score Against',
          'Expected Goals','Avg Shot Dist','Margin'
          ,'Settled Goals agaisnt' , 'Save %']]
    
    t4 = imp.fit_transform(t3)
    t4 = stdscaler.fit_transform(t4)
    
    
    preds = model.predict(t4)
    
    t3['predictions'] = preds
    
    
    return t3['predictions'].sum()

def projectfinal(initial, metric1 , avg1 , metric2, avg2, metric3 , avg3):
    
    t1 = replace(initial , metric1 , avg1)
    t2 = replace(t1 , metric2 , avg2)
    t3 = replace(t2, metric3 , avg3)
    
    t3 = t3.drop(columns = ['Season' , 'Week' , 'Game' , 'Team' , 'Opponent'])
    t3 = t3[['Shots','Shot%','Efficiency','Possession %',
          'Settled Goal','FB Goals','Assisted Goals','Shot Quality Ratio', 
          'D Efficiency','Turnovers', 'Score Against',
          'Expected Goals','Avg Shot Dist','Margin'
          ,'Settled Goals agaisnt' , 'Save %']]
    
    t4 = imp.fit_transform(t3)
    t4 = stdscaler.fit_transform(t4)
    
    
    preds = model.predict(t4)
    
    t3['predictions'] = preds
    
    
    return t3    