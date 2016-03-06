import numpy as np;
import math
import pandas as pd;
from pprint import pprint;
import csv
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
import logging





logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
stemmer = SnowballStemmer('english')
cached_stop_words = stopwords.words("english")
lmn = WordNetLemmatizer()








def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_lemm(s):
    return " ".join([lmn.lemmatize(w) for w in s.lower().split() if not w in cached_stop_words])


def preprocess_input_data(lemmatize = 1, remove_numbers = 1, remove_no_chars = 1, save_to_file = 1):
    '''
        PREPROCESSING
    '''  
    logging.log(logging.INFO, 'Preprocessing input data - start')
    train_data_raw = 'data/train.csv'
    test_data_raw = 'data/test.csv'
    desc = 'data/product_descriptions.csv'
    
    
    train_data = pd.read_csv(train_data_raw, encoding="ISO-8859-1")
    test_data = pd.read_csv(test_data_raw, encoding="ISO-8859-1")
    descs = pd.read_csv(desc)
    

    
    train_data['title_len'] = train_data['product_title'].map(lambda x: len(x.split()))
    test_data['title_len'] = test_data['product_title'].map(lambda x: len(x.split()))
    logging.log(logging.INFO, '\ttitle len')
    
    
    if remove_numbers:
        train_data['product_title'] = train_data['product_title'].map(lambda x : re.sub(r"(\d+)", "NUMBER", x))
        test_data['product_title'] = test_data['product_title'].map(lambda x : re.sub(r"(\d+)", "NUMBER", x))
        logging.log(logging.INFO, '\tproduct_title - numbers removed')
        
        train_data['search_term'] = train_data['search_term'].map(lambda x : re.sub(r"(\d+)", "NUMBER", x))
        test_data['search_term'] = test_data['search_term'].map(lambda x : re.sub(r"(\d+)", "NUMBER", x))
        logging.log(logging.INFO, '\tsearch term numbers removed')
    
    
    if remove_no_chars:    
        train_data['product_title'] = train_data['product_title'].map(lambda x : re.sub("[^a-zA-Z#@]", " ", x))
        test_data['product_title'] = test_data['product_title'].map(lambda x : re.sub("[^a-zA-Z#@]", " ", x))
        logging.log(logging.INFO, '\tproduct title no chars removed')
        
        train_data['search_term'] = train_data['search_term'].map(lambda x : re.sub("[^a-zA-Z#@]", " ", x))
        test_data['search_term'] = test_data['search_term'].map(lambda x : re.sub("[^a-zA-Z#@]", " ", x))
        logging.log(logging.INFO, '\tsearch term no chars removed')
        
    
    if lemmatize:
        #train_data['product_title'] = train_data['product_title'].map(lambda x: str_stemmer(x))
        train_data['product_title'] = train_data['product_title'].map(lambda x: str_lemm(x))
        #test_data['product_title'] = test_data['product_title'].map(lambda x: str_stemmer(x))
        test_data['product_title'] = test_data['product_title'].map(lambda x: str_lemm(x))
        logging.log(logging.INFO, '\tproduct title lemmatized')
    
        #train_data['search_term'] = train_data['search_term'].map(lambda x : str_stemmer(x))
        #test_data['search_term'] = test_data['search_term'].map(lambda x : str_stemmer(x))
        train_data['search_term'] = train_data['search_term'].map(lambda x : str_lemm(x))
        test_data['search_term'] = test_data['search_term'].map(lambda x : str_lemm(x))
        logging.log(logging.INFO, '\tsearch_term lemmatized')

    
    train_data['all_info'] = train_data['search_term'] + ' ' + train_data['product_title']
    test_data['all_info'] = test_data['search_term'] + ' ' + test_data['product_title']
    logging.log(logging.INFO, '\tall info created')
 
    
    train_data = pd.merge(train_data, descs, how='left', on='product_uid')
    test_data = pd.merge(test_data, descs, how='left', on='product_uid')
    logging.log(logging.INFO, '\tattribs loaded')

    if remove_numbers:
        train_data['product_description'] = train_data['product_description'].map(lambda x : re.sub(r"(\d+)", "NUMBER", x))
        test_data['product_description'] = test_data['product_description'].map(lambda x : re.sub(r"(\d+)", "NUMBER", x))
        logging.log(logging.INFO, '\tproduct desc numbers removed')
        
    if remove_no_chars:
        train_data['product_description'] = train_data['product_description'].map(lambda x : re.sub("[^a-zA-Z#@]", " ", x))
        test_data['product_description'] = test_data['product_description'].map(lambda x : re.sub("[^a-zA-Z#@]", " ", x))
        logging.log(logging.INFO, '\tproduct desc no chars removed')
        
    if lemmatize:
        #train_data['product_description'] = train_data['product_description'].map(lambda x: str_stemmer(x))
        #test_data['product_description'] = test_data['product_description'].map(lambda x : str_stemmer(x))
        train_data['product_description'] = train_data['product_description'].map(lambda x: str_lemm(x))
        test_data['product_description'] = test_data['product_description'].map(lambda x : str_lemm(x))
        logging.log(logging.INFO, '\tproduct desc lemmatized')
        
    
    train_data['all_info'] = train_data['all_info'] + ' ' + train_data['product_description']
    test_data['all_info'] = test_data['all_info'] + ' ' + test_data['product_description']
    logging.log(logging.INFO, '\tall info updated')

    
    train_data = train_data.drop(['search_term', 'product_title', 'product_uid', 'product_description'], axis=1)
    test_data = test_data.drop(['search_term', 'product_title', 'product_uid', 'product_description'], axis=1)
    
    logging.log(logging.INFO, 'Preprocessing input data - done')
    #pprint(train_data.head(5))
    
    if save_to_file:
        with open('preprocessed_train.data', 'wb') as prepr_data:
            cPickle.dump(train_data, prepr_data)
        
        with open('preprocessed_test.data', 'wb') as prepr_data2:
            cPickle.dump(test_data, prepr_data2)
    
        logging.log(logging.INFO, 'Preprocessed input data saved')




'''

0,4401600745628


'''


#print(train_data.shape)
#print(test_data.shape)
#pprint(train_data.columns)

load_saved_data = 0
final_run = 1




if load_saved_data == 0:
    preprocess_input_data()
    


with open('preprocessed_train.data') as prepr_data:
    train_data = cPickle.load(prepr_data)
    
with open('preprocessed_test.data') as prepr_data2:
    test_data = cPickle.load(prepr_data2)



vectorizer = TfidfVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None,
                                     max_features=5000, ngram_range=(1,2), encoding='utf-8',decode_error='ignore')

rmses = []




if final_run == 0:
    
    kf = KFold(train_data.shape[0], n_folds=10)
    for train, test in kf:
        
        logging.log(logging.INFO, 'starting iteration')
        X_train, X_test, y_train, y_test = train_data['all_info'].ix[train], train_data['all_info'].ix[test], train_data['relevance'].ix[train], train_data['relevance'].ix[test]
        
        
        X_lengths = np.array(train_data['title_len'].ix[train])
        Y_lengths = np.array(train_data['title_len'].ix[test])
        

        
    
        train_data_features = vectorizer.fit_transform(X_train)
        train_data_features = train_data_features.toarray()
        vocab = vectorizer.get_feature_names()
        
        
        #print('train_data_features shape = %s, type = %s' % (train_data_features.shape, type(train_data_features)))
        #print('X_lengths shape = %s, type = %s' % (X_lengths.shape, type(X_lengths)))
        
        #train_data_features = np.hstack((train_data_features, X_lengths))
        #train_data_features = np.concatenate((train_data_features, X_lengths), axis=1)
        train_data_features = np.column_stack((train_data_features, X_lengths))
        #pprint(train_data_features.shape)
        #quit()
        
        
        #adding title
        
        #train_features = np.hstack((train_features, train_features_unig))
        
        test_data_features = vectorizer.transform(X_test)
        test_data_features = test_data_features.toarray()
        
        #print('train_data_features shape = %s, type = %s' % (test_data_features.shape, type(test_data_features)))
        #print('X_lengths shape = %s, type = %s' % (Y_lengths.shape, type(Y_lengths)))
        
        test_data_features = np.column_stack((test_data_features, Y_lengths))
        
        #print('Train data shapes = %s, test data shapes = %s' % (train_data_features.shape, test_data_features.shape))
        #pprint(vocab)
        
        regr = RandomForestRegressor(n_estimators=10, max_depth=6, random_state=0, n_jobs=6)
        clf = BaggingRegressor(regr, n_estimators=20, max_samples=0.1, random_state=25, n_jobs=6)
        
        clf.fit(train_data_features, y_train)
        #print('trained')
        
        result = clf.predict(test_data_features)
        #print('prediction ready')
        
        #scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=5)
        #pprint(scores)
        #pprint(result.shape)
        #pprint(y_test.shape)
        
        tst = zip(result, y_test)
        
        errors = 0.0
        
        for row in tst:
            #print(row)
            diff = math.sqrt((row[0] - row[1]) * (row[0] - row[1]))
            errors += diff
            
            
        RMSE = errors / len(tst)
    
        print('RMSE = %s' % (RMSE))
        rmses.append(RMSE) 
        print('Running AVG RMSE = %s (%s)' % (sum(rmses) / len(rmses), len(rmses)))
        
        
        
        
    
    
    pprint(rmses)
    pprint('AVG RMSE = %s' % (sum(rmses)/10.0))
    '''
        AVG RMSE =  0.437510198403, MIN = 0.4225357693683969, MAX = 0.44258528972494987
        
        against test data RMSE = 0.53175
    '''
else:
    #final_run == 1
    X_train = train_data['all_info']
    y_train = train_data['relevance']
    X_test = test_data['all_info']
    
    X_lengths = np.array(train_data['title_len'])
    Y_lengths = np.array(test_data['title_len'])
    
    train_data_features = vectorizer.fit_transform(X_train)
    train_data_features = train_data_features.toarray()
    
    train_data_features = np.column_stack((train_data_features, X_lengths))
    
    test_data_features = vectorizer.transform(X_test)
    test_data_features = test_data_features.toarray()
    
    test_data_features = np.column_stack((test_data_features, Y_lengths))
    
    regr = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=0, verbose=3, n_jobs=6)
    clf = BaggingRegressor(regr, n_estimators=20, max_samples=0.5, random_state=25, n_jobs=6)
        
    clf.fit(train_data_features, y_train)
    
    result = clf.predict(test_data_features)
    
    #pprint(len(result))
    #pprint(X_test.shape)
    #pprint(test_data.shape)
    
    output = zip(test_data['id'], result)
    
    output = pd.DataFrame({"id" : test_data['id'], "relevance" : result}).to_csv('out.csv', index=False)
    print('Done')
    

    
  