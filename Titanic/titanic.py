import numpy as np;
import math
import pandas as pd;
from pprint import pprint;
import csv
from sklearn.ensemble import RandomForestClassifier

'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import cPickle
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
'''
import logging


training_data_raw = 'data/train.csv'
testing_data_raw = 'data/test.csv'


train_data = pd.read_csv(training_data_raw, encoding="ISO-8859-1")
test_data = pd.read_csv(testing_data_raw, encoding="ISO-8859-1")


#pprint(train_data.columns)
'''
Index([u'PassengerId', u'Survived', u'Pclass', u'Name', u'Sex', u'Age',
       u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked'],
      dtype='object')
'''
'''
   PassengerId  Survived  Pclass        Name                                                   Sex      Age  SibSp  Parch        Ticket                  Fare      Cabin Embarked
0            1         0       3        Braund, Mr. Owen Harris                                male     22      1     0          A/5 21171               7.2500    NaN        S
1            2         1       1        Cumings, Mrs. John Bradley (Florence Briggs Th...      female   38      1     0          PC 17599                71.2833   C85        C
2            3         1       3        Heikkinen, Miss. Laina                                 female   26      0     0          STON/O2.    3101282     7.9250    NaN        S

pprint(train_data.head(10))
'''


#selected columns for prediction - PClass, Sex, Age, SibSp, Parch, Cabin, Embarked
train_data = train_data.drop(['PassengerId', 'Name'], axis=1)
test_data = test_data.drop(['Name'], axis=1)


X_train = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
Y_train = train_data['Survived']


pprint(train_data.columns)
pprint(Y_train.shape)
pprint(X_train.columns)
pprint(X_train.shape)

def convert_sex (str):
    if str == 'male':
        return(1)
    else:
        return(0)
    
    
def convert_age(str):
    if np.isnan(str):
        return(0.0)
    else:
        return(float(str))
    
# test_data['product_description'] = test_data['product_description'].map(lambda x : str_lemm(x))
        
X_train['SexBit'] = X_train['Sex'].map(lambda x : convert_sex(x))
X_train = X_train.drop(['Sex'], axis=1)
X_train['Age'] = X_train['Age'].map(lambda x : convert_age(x))

X_test['SexBit'] = X_test['Sex'].map(lambda x : convert_sex(x))
X_test = X_test.drop(['Sex'], axis=1)
X_test['Age'] = X_test['Age'].map(lambda x : convert_age(x))



pprint(X_train.head(10))



pred = RandomForestClassifier(n_estimators=5000, n_jobs=6, verbose=3)
pred.fit(X_train, Y_train)

result = pred.predict(X_test)

#pprint(result)
#pprint(len(result))
#pprint(test_data)

 
output = pd.DataFrame({'PassengerId' : test_data['PassengerId'], 'Survived' : result}).to_csv('titanic_out.csv', index=False)  
   




print('Done')