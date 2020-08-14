import torch
import dill
import pandas as pd
from random import random
from operator import itemgetter
from electra_spacing.tokenizer import get_tokenizer


special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

class SpacingDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        file_path,
        tokenizer=None,
        seq_len=128,
        padding_idx=0,
        use_padding = True,
        threshold = 0.6):
        
        if tokenizer is None:
            self.tokenizer = get_tokenizer()
        else:
            self.tokenizer = tokenizer
        
        self.threshold = threshold
        self.seq_len = seq_len
        self.pad_token_id = self.tokenizer.pad_token_id
        self.use_padding = use_padding

        if 'txt' in file_path:
            self.input_text = []
            lines = open(file_path, encoding="utf-8").readlines()

            for l in lines:
                l = l.replace('\n', '').strip()
                if len(l) >= minimum_size:
                    self.input_text.append(l)
        elif 'tsv' in file_path:
            lines = pd.read_csv(file_path, sep='\t', header=None)
            self.input_text = lines[0].tolist()
        
        self.len = len(self.input_text)

    def tokenize(self, text: str, padding: bool = True, return_tensor: bool = True):
            
        tokens = self.tokenizer.encode(text)
        ##consider single token only
        segment_ids = [0] * len(tokens)

        if type(tokens) == list:
            tokens = torch.tensor(tokens)
            
        if padding:
            if len(tokens) >= self.seq_len:
                tokens = tokens[:self.seq_len]
                segment_ids = torch.tensor(segment_ids[:self.seq_len])
            else:
                pad_tensor = torch.tensor(
                    [self.pad_token_id] * (self.seq_len - len(tokens))
                )
                tokens = torch.cat((tokens, pad_tensor), 0)
                segment_ids = torch.tensor([0] * self.seq_len)

        if return_tensor:
            return (tokens, segment_ids)
        else:
            return (tokens.numpy(), segment_ids.numpy())    
            
    def __getitem__(self, idx):
        sentence = self.input_text[idx]
        new_sentence = ''
        for char in sentence:
            if random() < self.threshold and char == ' ':
                pass
            else:
                new_sentence += char

        (tokens, token_type_ids) = self.tokenize(text=new_sentence)
        (labels, _) = self.tokenize(text=sentence)
        
        labels_weight = [0] * self.seq_len
        label_token = []
        
        for l in labels:    
            label_token.append(self.tokenizer.ids_to_tokens[l.item()])
            
        for i, token in enumerate(label_token):
            if token not in special_tokens and '##' not in token:
                labels_weight[i] = 1
                
        labels_weight = torch.tensor(labels_weight)
        
        return (tokens, token_type_ids, labels, labels_weight)

    def __len__(self):
        return self.len

