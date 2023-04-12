import pandas as pd
import numpy as np
import time,sys
from pathlib import Path
# text preprocessing
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from scipy.special import softmax
from word_embedding import flair_embedding_options,load_word_embedding
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
import joblib
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression,SGDClassifier, LinearRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
# feature extraction / vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from collect_info import collect_best


def choose_clf(clf='MLP'):
    models = {
    'SVM':make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3)),
    'MLP':MLPClassifier(alpha=0.01, batch_size=256, 
        epsilon=1e-08, hidden_layer_sizes=(300,), 
        learning_rate='adaptive', max_iter=500),
    'KNN':KNeighborsClassifier(n_neighbors=3,n_jobs=-1),
    'LR':LogisticRegression(random_state=0,multi_class='multinomial',n_jobs=-1,class_weight='balanced'),
    'ExtraTrees':ExtraTreesClassifier(n_estimators=100, random_state=0,n_jobs=-1,class_weight='balanced'),
    'RandomForest':RandomForestClassifier(random_state=0,n_jobs=-1,class_weight='balanced'),
    'BernoulliNB':BernoulliNB(),
    'GaussianNB':GaussianNB()
    }
    return models[clf]


def preprocess_and_tokenize(data):    

    #remove html markup
    data = re.sub("(<.*?>)", "", data)

    #remove urls
    data = re.sub(r'http\S+', '', data)
    
    #remove hashtags and @names
    data= re.sub(r"(#[\d\w\.]+)", '', data)
    data= re.sub(r"(@[\d\w\.]+)", '', data)

    #remove punctuation and non-ascii digits
    data = re.sub("(\\W|\\d)", " ", data)
    
    #remove whitespace
    data = data.strip()
    
    # tokenization with nltk
    data = word_tokenize(data)
    
    # stemming with nltk
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]
        
    return stem_data

def prepare_train_and_test(dataset='',feat='tfidf',resampling='',
        development=True):  
    data_path = f'data/{dataset}'
    df_train = pd.read_csv(data_path+'_train.csv')
    df_dev = pd.read_csv(data_path+'_dev.csv')


    X_train = df_train.text.tolist()
    y_train = df_train.hard_label.tolist()
    if development==True:
        X_train,X_test,y_train,y_test = train_test_split(X_train,y_train, 
            test_size= 0.3, stratify=y_train, random_state=42)
        # X_test = df_dev.text.tolist()
        # y_test = df_dev.hard_label.tolist()
    else:
        X_test = df_dev.text.tolist()
        y_test = df_dev.hard_label.tolist()
    
    if feat == 'tfidf':
        # TFIDF, unigrams and bigrams
        vect = TfidfVectorizer(tokenizer=preprocess_and_tokenize, 
            sublinear_tf=True, norm='l2', ngram_range=(1, 2),max_features=3000)
        # fit on our complete corpus
        data = X_train+X_test
        vect.fit(data)
        X_train = np.array(vect.transform(X_train).todense())
        X_test = np.array(vect.transform(X_test).todense())

    elif feat in flair_embedding_options+['ar']:
        document_embedding = load_embedding_model(embedding_option=feat)
        X_train = fit_corupus(X_train,document_embedding)
        X_test = fit_corupus(X_test,document_embedding)
    else:
        print("option not validated in the system")

    if resampling =='+over':
        sampler = SMOTE(random_state=0,n_jobs=-1)
    elif resampling == '+under':
        sampler = NearMiss(n_jobs=-1)
        # sampler = RandomUnderSampler(random_state=42)
        # sampler = ClusterCentroids(n_jobs=-1,random_state=42)
        # sampler = TomekLinks(ratio='not minority',n_jobs=-1)
    elif resampling == '+comb':
        #perform over-sampling using SMOTE and cleaning using Tomek links.
        sampler = SMOTETomek(sampling_strategy='auto',random_state=0)
    
    if resampling !='':
        X_train,y_train = sampler.fit_resample(X_train,y_train)

    
    return X_train, y_train, X_test, y_test

def load_embedding_model(embedding_option):
    embedding_model = WordEmbeddings(embedding_option)
    document_embedding = DocumentPoolEmbeddings([embedding_model])
    return document_embedding

def fit_corupus(sentences,document_embedding):
    fitted = []
    for doc in sentences:
        feat = load_word_embedding(doc,document_embedding)
        fitted.append(feat)
    return np.asarray(fitted)

def train_data(X_train,y_train,X_test,y_test,dataset='',clf='RandomForest',
    feat='tfidf',sample_weight=[],weight_method='',development=True):
    model=choose_clf(clf)
    start = time.time()
    if len(sample_weight)>0:
        model.fit(X_train,y_train,sample_weight=sample_weight)
        clf = clf+'+'+weight_method

    else:
        model.fit(X_train,y_train)
    train_dur = time.time()-start
    
    start = time.time()
    y_pred=model.predict(X_test)
    test_dur = time.time() - start
    if development == False:
        joblib.dump(model,f'saved_model/{dataset}_{feat}-{clf}.pkl',compress=3)
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    b_accuracy = balanced_accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test,y_pred=y_pred,average = 'micro')
    print(f'{feat}, {clf}')
    print("Accuracy: {:.2f}%".format(accuracy*100))
    print("F1: {:.2f}%".format(f1*100))
    # report = classification_report(y_test, y_pred,output_dict=True)
    # df = pd.DataFrame(report).transpose()
    # df.to_csv(f'reports/{dataset}_{feat}-{clf}.csv',index=True)
    # cm = confusion_matrix(y_test, y_pred,normalize='true')
    # ConfusionMatrixDisplay(confusion_matrix=cm).plot(xticks_rotation='vertical')

    # plt.title(dataset+f'_{feat}-{clf}')
    # plt.tight_layout()
    # plt.savefig(f'figs/{dataset}_{feat}-{clf}.png')
    return accuracy,b_accuracy,f1,train_dur,test_dur



def find_best_model(method='baseline'):
    for current_dataset in ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit']:
        feats = flair_embedding_options+['tfidf'] if current_dataset!='ArMIS' \
            else ['ar','tfidf'] #
        test_models = ['GaussianNB','BernoulliNB','MLP','LR','RandomForest','ExtraTrees',
                'SVM','KNN']

        for feat in feats:
            X_train,y_train, X_test, y_test =prepare_train_and_test(
                dataset=current_dataset,feat=feat)
            if method!='baseline':
                data_path = f'data/{current_dataset}'
                df_train = pd.read_csv(data_path+'_train.csv')
                sample_weight = df_train[method].tolist()
                sample_weight,_ = train_test_split(sample_weight, 
                test_size= 0.3, random_state=42)
                test_models = ['GaussianNB','BernoulliNB','LR',
                'RandomForest','ExtraTrees']
                # MLP,SVM,KNN don't have sample weight

            results = []
            for test_model in test_models:
                if method!='baseline':
                    acc, b_acc, f1, train_dur, test_dur = \
                    train_data(X_train,y_train,X_test,y_test,
                        dataset=current_dataset,clf=test_model,feat=feat,
                        sample_weight=sample_weight,weight_method=method)
                else:
                    acc, b_acc, f1, train_dur, test_dur = \
                        train_data(X_train,y_train,X_test,y_test,
                        dataset=current_dataset,clf=test_model,feat=feat)
                
                results.append({'model':test_model,'acc':acc,
                    'b_acc':b_acc, 'f1':f1,
                    'train_dur':train_dur,
                    'test_dur':test_dur})
            df = pd.DataFrame(results)
            df.to_csv(f'reports/log/{current_dataset}_{feat}_{method}.csv',index=False)
    pass

def train_and_save_best_model(method='baseline',metrics='f1'):
    for current_dataset in ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit']:
        print('******'+current_dataset)
        df = pd.read_csv(f'reports/summary/{current_dataset}_{method}_{metrics}.csv')
        best = df.model.to_list()[0]
        feat = best.split('-')[0]
        clf = best.split('-')[1]
        X_train,y_train, X_test, y_test =prepare_train_and_test(
                dataset=current_dataset,feat=feat,development=False)
        if method!='baseline':
            data_path = f'data/{current_dataset}'
            df_train = pd.read_csv(data_path+'_train.csv')
            sample_weight = df_train[method].tolist()
            train_data(X_train,y_train,X_test,y_test,
                dataset=current_dataset,clf=clf,feat=feat,
                sample_weight=sample_weight,weight_method=method,
                development=False)
        else:
            train_data(X_train,y_train,X_test,y_test,
                dataset=current_dataset,clf=clf,feat=feat,
                development=False)

    pass
# def load_saved_sets(feat):
#     X_train = hkl.load(f'processed_data/X_train_{feat}.hkl')
#     X_test = hkl.load(f'processed_data/X_test_{feat}.hkl')
#     y_train = pickle.load(open(f'processed_data/y_train.hkl','rb'))
#     y_test = pickle.load(open(f'processed_data/y_test.hkl','rb'))
#     return X_train,y_train,X_test,y_test 


# y = np.clip(y, 1e-8, 1 - 1e-8)   # numerical stability
# inv_sig_y = np.log(y / (1 - y))  # transform to log-odds-ratio space

# from sklearn.linear_model import LinearRegression
# lr = LinearRegression()
# lr.fit(X, inv_sig_y)


# # we can input soft labels 
# def sigmoid(x):
#     ex = np.exp(x)
#     return ex/(1-ex)


# preds = sigmoid(lr.predict(X_new))


if __name__ == '__main__':
    # find_best_model()
    #let's add AnnoSoft
    # find_best_model(method='anno_soft')
    # find_best_model(method='anno_hard')
    # collect_best()
    train_and_save_best_model()
    train_and_save_best_model(method='anno_soft')
    train_and_save_best_model(method='anno_hard')