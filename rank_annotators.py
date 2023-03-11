import pandas as pd

# what to get:
# 1. how many annotators participate in the tasks
# 2. for each annotator, how many tasks and dataset they submit results
# 3. how many submissions are actually the hard label

# df = pd.read_csv('annotators.csv')

# # total number of unique annotators

# unique_annotators = df['annotator'].unique()
# print(len(unique_annotators))

# groups = df.groupby("dataset")['annotator'].unique()
# print(groups)

def collect_annotator_info():
    df = pd.read_csv('annotators.csv')
    grouped = df.groupby("dataset")

    for dataset, group_data in grouped:
        annotator_scores = {}
        for row in group_data.to_dict('records'):

            if row['annotator'] not in annotator_scores.keys():
                annotator_scores[row['annotator']] = {}
                annotator_scores[row['annotator']]['score']=0
                annotator_scores[row['annotator']]['correct']=0
                annotator_scores[row['annotator']]['num_task']=1
            else:
                annotator_scores[row['annotator']]['num_task']+=1
            if row['annotator_label']==row['hard_label']:
                annotator_scores[row['annotator']]['score']+=max(row['soft_label_0'],row['soft_label_1'])
                annotator_scores[row['annotator']]['correct']+=1
        df = pd.DataFrame(annotator_scores)
        print(dataset,df)
        df.to_csv('annotator_res/'+dataset+'annotator_score.csv')
    pass


# compute sample weight by averaging AnnoSoft
# AnnRnk and AnnoHard is computed by the annotator info: 
# score, num_correct, num_task
# score = sum(max(soft_label_0,soft_label_1))
def compute_AnnoSoft(score,num_correct):
    if num_correct == 0:
        return 0
    AnnoSoft = float(score)/float(num_correct)
    return AnnoSoft

def compute_AnnoHard(num_correct,num_task):
    if num_correct == 0:
        return 0
    AnnoHard = float(num_correct)/float(num_task)
    return AnnoHard

def add_computed_values(dataset):
    df = pd.read_csv('annotator_res/'+dataset+'annotator_score.csv',
        index_col=0)
    anno_dict = df.to_dict()
    # print(anno_dict)
    for anno in anno_dict.keys():
        anno_soft = compute_AnnoSoft(anno_dict[anno]['score'],
            anno_dict[anno]['correct'])
        anno_hard = compute_AnnoHard(anno_dict[anno]['correct'],
            anno_dict[anno]['num_task'])
        anno_dict[anno]['anno_soft']=anno_soft
        anno_dict[anno]['anno_hard'] = anno_hard
    df = pd.DataFrame(anno_dict)
    print(dataset,df)
    df.to_csv('annotator_res/'+dataset+'annotator_score.csv')
    pass

def compute_sample_weight(dataset='', weight_method = 'anno_soft'):
    data_path = f'data/{dataset}'
    df_train = pd.read_csv(data_path+'_train.csv')
    anno_list = df_train.annotators.to_list()
    df = pd.read_csv('annotator_res/'+dataset+'annotator_score.csv',
        index_col=0)
    anno_dict = df.to_dict()
    
    sample_weights = []
    for annotators in anno_list:
        sample_weight = 0
        annotators = annotators.split(',')
        for anno in annotators:
            sample_weight += anno_dict[anno][weight_method]
        sample_weight = sample_weight/float(len(annotators))
        sample_weights.append(sample_weight)
    df_train[weight_method] = sample_weights
    df_train.to_csv(data_path+'_train.csv',index=False)
    pass


if __name__ == '__main__':
    for current_dataset in ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit']:
    #     add_computed_values(current_dataset)
        # weight_method = 'anno_soft'
        weight_method = 'anno_hard'
        compute_sample_weight(dataset=current_dataset,
            weight_method=weight_method)