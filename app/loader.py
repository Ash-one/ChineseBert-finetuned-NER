import torch
import os
from fastNLP.io import DataBundle
from fastNLP import DataSet, Instance
from fastNLP.transformers.torch import BertTokenizer
from fastNLP import cache_results, Vocabulary
from fastNLP import prepare_torch_dataloader

def text2dataset(text:str):
    ds = DataSet()
    if text != '':  
        ds.append(Instance(raw_words = list(text)))
    return ds

def process_predict_data(data_bundle, model_name):

    tokenizer = BertTokenizer.from_pretrained(model_name)
    def bpe(raw_words):
        bpes = [tokenizer.cls_token_id]
        first = [0]
        first_index = 1  # 记录第一个bpe的位置
        for word in raw_words:
            bpe = tokenizer.encode(word, add_special_tokens=False)
            bpes.extend(bpe)
            first.append(first_index)
            first_index += len(bpe)
        bpes.append(tokenizer.sep_token_id)
        first.append(first_index)
        return {'input_ids': bpes, 'input_len': len(bpes), 'first': first, 'seq_len': len(raw_words)}
    # 对data_bundle中每个dataset的每一条数据中的raw_words使用bpe函数，并且将返回的结果加入到每条数据中。
    data_bundle.apply_field_more(bpe, field_name='raw_words', num_proc=1)

    return data_bundle, tokenizer

def text2dataloader(text:str):
    '''
    输入待预测的文本，返回dataloader和tokenizer
    '''
    predict_data_bundle = DataBundle(datasets={
        "predict": text2dataset(text),
    })
    predict_data_bundle, predict_tokenizer = process_predict_data(predict_data_bundle, 'hfl/rbt3')
    predict_dataloaders = prepare_torch_dataloader(predict_data_bundle, batch_size=1)

    return predict_dataloaders, predict_tokenizer

def read_list_from_text(path):
    '''
    从txt文件读取label—idx对应的列表
    '''
    final_list = []
    with open(path,'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip().split(',')
            final_list.append((line[0],int(line[1])))
    return final_list

def load_model_from_file(name:str):
    name = 'app/static/models/' + name

    if os.path.exists(name+'.pth') and os.path.exists(name+'.txt'):
        # 在服务器上没有gpu,只使用cpu加载
        model = torch.load(name + '.pth',map_location=torch.device('cpu'))
        label_idx_list = read_list_from_text(name + '.txt')
        return model, label_idx_list
    else:
        print(name,'Model not found')
        return None, None

def predict(text:str, model):
    '''
    输入待预测的文本和模型，返回预测结果
    '''
    predict_dataloaders, predict_tokenizer = text2dataloader(text)
    dev = next(model.parameters()).device
    model.eval()

    for data in predict_dataloaders['predict']:
        input_ids = torch.LongTensor(data['input_ids']).to(dev)
        input_len = torch.LongTensor(data['input_len']).to(dev)
        first =     torch.LongTensor(data['first']).to(dev)
        
        result = model.evaluate_step(input_ids,input_len,first)['pred']
    
    return result[-1]

def convert_result2label(result, label_idx_list):
    '''
    将预测结果转换为标签
    '''
    final_result = []
    for idx in result:
        final_result.append(label_idx_list[idx][0])
    return final_result

def convert_label2entity(labels):
    '''
    将标签转换为实体,由于只有BI两个标签，并且存在实体嵌套的可能，所以I标签也可以做为开始标签
    return [{'start':0,'end':3,'label':'ORG'},{'start':4,'end':6,'label':'PER'}]
    '''
    final_result = []
    entity = {'label':"O"}
    for idx,label in enumerate(labels):
        if label[0] == 'B':
            entity['start'] = idx
            entity['label'] = label[2:]
        elif label[0] == 'I':
            if idx == len(labels)-1:
                final_result.append(entity)
                entity = {'label':"O"}
            elif labels[idx+1][0] != 'I'  :
                entity['end'] = idx+1
                final_result.append(entity)
                entity = {'label':"O"}
            elif labels[idx+1][2:] != entity['label']:
                entity['end'] = idx+1
                final_result.append(entity)
                entity = {'label':labels[idx+1][2:],'start':idx+1}
    return final_result

def load_colors_preset():
    '''
    颜色和实体的对应关系，使用displacy要求的规范
    '''
    colors = {
        "ORG.NAM": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
        "GPE.NAM": "linear-gradient(90deg, #BE5869, #7b4397)",
        "PER.NAM": "linear-gradient(90deg, #c2e59c, #64b3f4)",
        "LOC.NAM": "linear-gradient(90deg, #8E0E00, #1F1C18)",
        "LOC.NOM": "linear-gradient(90deg, #00C9FF, #92FE9D)",
        "PER.NOM": "linear-gradient(90deg, #fc00ff, #00dbde)",
        "ORG.NOM": "linear-gradient(90deg, #9cfce7, #9cfce7)",
    }
    return colors