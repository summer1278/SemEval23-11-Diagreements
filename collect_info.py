import json
import pandas as pd

# get train dev split
def train_dev_split():
  header = ["dataset","split","id",
          "hard_label","soft_label_0","soft_label_1","text","lang","annotators"]
  for current_dataset in ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit']:
    for current_split in ['train','dev']:
      results = []
      current_file = './'+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json'
      data = json.load(open(current_file,'r', encoding = 'UTF-8'))
      for item_id in data:
        text = data[item_id]['text']        
        text = text.replace('\t',' ').replace('\n',' ').replace('\r',' ')
        hard_label = data[item_id]['hard_label']
        soft_label_0 = data[item_id]['soft_label']["0"]
        soft_label_1 = data[item_id]['soft_label']["1"]
        lang = data[item_id]['lang']
        annotators = data[item_id]['annotators']
        results.append([current_dataset, current_split, 
          item_id,hard_label,soft_label_0,soft_label_1,
          text,lang,annotators])
      new_df = pd.DataFrame(results,columns=header)
      new_df.to_csv(f'data/{current_dataset}_{current_split}.csv',index=False)
  pass


# get test split
def test_split():
  header = ["dataset","split","id",
          "hard_label","soft_label_0","soft_label_1","text","lang"]
  for current_dataset in ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit']:                        # loop on datasets
    results = []
    for current_split in ['test']:
      current_file = './data_evaluation/'+current_dataset+'_'+current_split+'.json'
      data = json.load(open(current_file,'r', encoding = 'UTF-8'))
      for item_id in data:
        text = data[item_id]['text']        
        text = text.replace('\t',' ').replace('\n',' ').replace('\r',' ')
        hard_label = data[item_id]['hard_label']
        soft_label_0 = data[item_id]['soft_label']["0"]
        soft_label_1 = data[item_id]['soft_label']["1"]
        lang = data[item_id]['lang']
        results.append([current_dataset, current_split, 
          item_id,hard_label,soft_label_0,soft_label_1,
          text,lang])
      new_df = pd.DataFrame(results,columns=header)
    new_df.to_csv(f'data/{current_dataset}_{current_split}.csv',index=False)
  pass


# get annotator info and labels
def split_by_annotators():
  header = ["dataset","split","id",
          "hard_label","soft_label_0","soft_label_1","annotator","annotator_label",
          "num_annotations"]
  results = []
  for current_dataset in ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit']:                        # loop on datasets
    for current_split in ['train']:
      current_file = './'+current_dataset+'_dataset/'+current_dataset+'_'+current_split+'.json'
      data = json.load(open(current_file,'r', encoding = 'UTF-8'))
      for item_id in data:
        annotators = data[item_id]['annotators'].split(',')
        annotations = data[item_id]['annotations'].split(',')
        num_annotations = data[item_id]["number of annotations"]
        hard_label = data[item_id]['hard_label']
        soft_label_0 = data[item_id]['soft_label']["0"]
        soft_label_1 = data[item_id]['soft_label']["1"]
        for ann,pred_label in zip(annotators,annotations):
          results.append([current_dataset, current_split, 
            item_id,hard_label,soft_label_0,soft_label_1,ann,pred_label,num_annotations])
  new_df = pd.DataFrame(results,columns=header)
  new_df.to_csv('annotators.csv',index=False)
  pass


def collect_method_results(current_dataset='',method='baseline',
  metrics= 'f1'):
  from glob import glob
  import os
  all_result_files = glob(f'reports/log/{current_dataset}_*_{method}.csv')
  # list_of_dataframes = [pd.read_csv(f) for f in all_result_files]
  list_of_dataframes = [] 
  for f in all_result_files:
    df = pd.read_csv(f)
    feat = os.path.basename(os.path.basename(f)).split('_')[1]
    df['model'] = f'{feat}-' + df['model'].astype(str)
    list_of_dataframes.append(df)
  df = pd.concat(list_of_dataframes)
  df = df.sort_values(by=[metrics,'b_acc'],
    ascending=False)
  print(df)
  df.to_csv(f'reports/summary/{current_dataset}_{method}_{metrics}.csv', index=False)
  pass


def collect_best():
  for current_dataset in ['ArMIS','MD-Agreement','ConvAbuse', 'HS-Brexit']:
    collect_method_results(current_dataset=current_dataset,method='baseline')
    collect_method_results(current_dataset=current_dataset,method='anno_rank')
    collect_method_results(current_dataset=current_dataset,method='anno_correct')
  pass
  pass

if __name__ == '__main__':
  # train_dev_split()
  # # test_split()
  # collect_best()
  pass

  

