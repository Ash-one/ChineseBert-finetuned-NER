import torch
from torch import nn
from fastNLP.transformers.torch import BertModel
from fastNLP import seq_len_to_mask
import torch.nn.functional as F
from fastNLP.modules.torch import ConditionalRandomField

class BertBilstmCrfNER(nn.Module):
    def __init__(self, model_name,num_class, embedding_dim = 768,hidden_size=512,dropout=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            num_layers=2,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_class)
        self.crf = ConditionalRandomField(num_class)
        

    def forward(self, input_ids, input_len,target=None):
        attention_mask = seq_len_to_mask(input_len)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        first_bpe_state = last_hidden_state[:, 1:-1]
        feats, _ = self.lstm(first_bpe_state) # 输入lstm
        feats = self.fc(feats)
        feats = self.dropout(feats)
        logits = F.log_softmax(feats, dim=-1)
        
        mask = seq_len_to_mask(input_len-2)
        
        if target is None:
            pred, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': pred}
        else:
            loss = self.crf(logits, target, mask).mean()
            return {'loss': loss}

    def train_step(self, input_ids, input_len, target):
        # {'loss':loss}
        return self(input_ids, input_len,target)

    def evaluate_step(self, input_ids, input_len,first):
        #  {'pred': pred}
        return self(input_ids, input_len)

class BertNER(nn.Module):
    '''
    Bert+MLP的NER模型
    '''
    def __init__(self, model_name, num_class):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.mlp = nn.Sequential(nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
                                nn.Dropout(0.3),
                                nn.Linear(self.bert.config.hidden_size, num_class))

    def forward(self, input_ids, input_len, first):
        attention_mask = seq_len_to_mask(input_len)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        first = first.unsqueeze(-1).repeat(1, 1, last_hidden_state.size(-1))
        first_bpe_state = last_hidden_state.gather(dim=1, index=first)
        first_bpe_state = first_bpe_state[:, 1:-1]  # 删除 cls 和 sep

        pred = self.mlp(first_bpe_state)
        return {'pred': pred}

    def train_step(self, input_ids, input_len, first, target):
        pred = self(input_ids, input_len, first)['pred']
        loss = F.cross_entropy(pred.transpose(1, 2), target)
        return {'loss': loss}

    def evaluate_step(self, input_ids, input_len, first):
        pred = self(input_ids, input_len, first)['pred'].argmax(dim=-1)
        return {'pred': pred}