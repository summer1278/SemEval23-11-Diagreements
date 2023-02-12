import pandas as pd 
import numpy as np
import joblib
from trad_train import load_embedding_model, preprocess_and_tokenize,fit_corupus
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn

def test(dataset,method='baseline',metrics='f1',split='dev'):
    data_path = f'data/{dataset}'
    df_test = pd.read_csv(data_path+f'_{split}.csv')
    X_test = df_test.text.tolist()

    df = pd.read_csv(f'reports/summary/{dataset}_{method}_{metrics}.csv')
    best = df.model.to_list()[0]
    feat = best.split('-')[0]
    clf = best.split('-')[1]
    if method!='baseline':
        model_name = f'saved_model/{dataset}_{feat}-{clf}+{method}.pkl'
    else:

        model_name = f'saved_model/{dataset}_{feat}-{clf}.pkl'
    model = joblib.load(model_name)
    if feat == 'tfidf':
        vect = TfidfVectorizer(tokenizer=preprocess_and_tokenize, 
                sublinear_tf=True, norm='l2', ngram_range=(1, 2),
                max_features=3000)
        # fit on our complete corpus
        df_train = pd.read_csv(data_path+'_train.csv')
        X_train = df_train.text.tolist()
        data = X_train+X_test
        vect.fit(data)
        X_test = np.array(vect.transform(X_test).todense())
        print(len(X_test))
    else:
        document_embedding = load_embedding_model(embedding_option=feat)
        X_test = fit_corupus(X_test,document_embedding)

    hard_pred=model.predict(X_test)
    soft_preds = model.predict_proba(X_test)
    return hard_pred,soft_preds

def write_to_truth_file(dataset,hard_pred,soft_preds,method='baseline',split='dev'):
    soft_pred_0, soft_pred_1 = zip(*soft_preds)
    file_name = f'res/{method}/{dataset}_results.tsv' \
        if split =='dev' \
            else f'res/submission/{dataset}_results.tsv'
    with open(file_name, 'w') as f:
        for a,b,c in zip(hard_pred,soft_pred_0,soft_pred_1):
            f.write(f"{a}\t{b}\t{c}\n")
        print('saved')
    pass

def get_data(myfile):
    soft = list()
    hard = list()
    with open(myfile,'r') as f:
            for line in f:
                line=line.replace('\n','')
                parts=line.split('\t')
                soft.append([float(parts[1]),float(parts[2])])
                hard.append(parts[0])
    return(soft,hard)


#===== snippet 4: function that calculates cross-entropy 

def cross_entropy(targets, predictions, epsilon = 1e-12):                                
    predictions = np.clip(predictions, epsilon, 1. - epsilon)                                      
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


#===== snippet 5: function that calculates weighted averaged f1

from sklearn.metrics import f1_score

def f1_metric(solution, prediction):
    f1_wa = sklearn.metrics.f1_score(solution, prediction, average = 'micro')                
    return f1_wa


def dev_submission():
    methods = ['baseline','anno_rank','anno_correct']
    for current_dataset in ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit']:  
        hard_scores=[]
        soft_scores=[]
        for method in methods:
            hard_pred,soft_preds = test(dataset=current_dataset,
                method=method,metrics='f1')
            write_to_truth_file(current_dataset,hard_pred,soft_preds,method=method)

            mytruthfile = f'res/{method}/{current_dataset}_results.tsv'

            soft_ref, hard_ref = get_data(mytruthfile)
            soft_pred, hard_pred = get_data(f'majority_baseline_practicephase/res/{current_dataset}_results.tsv')            # example of a result file

            soft_score = cross_entropy(soft_ref,soft_pred)
            hard_score = f1_metric(hard_ref,hard_pred)
            print(method,soft_score,hard_score)
            hard_scores.append(hard_score)
            soft_scores.append(soft_score)

        df = pd.DataFrame({'method':methods,
            'hard_score':hard_scores,
            'soft_score':soft_scores})
        df.to_csv(f'res/{current_dataset}_scores.csv',index=False)
    pass

def eval_submission(method):
    for current_dataset in ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit']:  
        hard_pred,soft_preds = test(dataset=current_dataset,
            method=method,metrics='f1',split='test')
        write_to_truth_file(current_dataset,hard_pred,soft_preds,split='test')
    pass

if __name__ == '__main__':
    eval_submission('baseline')
