'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Wrapped functions related to paddle for my repository: 
text-matching-explained (context-specific only)
'''

from os.path import exists
import json
from collections import defaultdict
from paddlenlp.datasets import MapDataset
from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Pad, Stack, Tuple
from paddlenlp.data import Vocab
from paddlenlp.transformers.bert.tokenizer import BasicTokenizer
from collections.abc import Iterable


def load_dataset(fpath, num_row_to_skip=0):
    def read(path):
        data = open(path)
        for _ in range(num_row_to_skip):
            next(data)
    
        for line in data:
            line = line.split('\t')
        
            yield line[0], line[1], int(line[2].rstrip())
    
    if isinstance(fpath, str):
        assert exists(fpath), f"{fpath} does not exist!"
        return list(read(fpath))
    
    elif isinstance(fpath, (list, tuple)):
        for fp in fpath:
            assert exists(fp), f"{fp} does not exist!"
        return [list(read(fp)) for fp in fpath]
    
    raise TypeError("Input fpath must be a (list) of valid filepath(es)")


def gather_text(dataset, end_col=2):
    out = []
    for data in dataset:
        out.extend(data[:end_col])
    return out


class TextVectorizer:
     
    def __init__(self, tokenizer=None):
        self.tokenize = tokenizer if tokenizer \
        else BasicTokenizer().tokenize
        self.vocab_to_idx = ''
        self.idx_to_vocab = ''
        self._V = None
    
    def build_vocab(self, text):
        tokens = list(map(self.tokenize, text))
        self._V = Vocab.build_vocab(tokens, unk_token='[UNK]', pad_token='[PAD]')
        self.vocab_to_idx = self._V.token_to_idx
        self.idx_to_vocab = self._V.idx_to_token
        
        print('Two vocabulary dictionaries have been built!\n' \
             + 'Please call \033[1mX.vocab_to_idx | X.idx_to_vocab\033[0m to find out more' \
             + ' where [X] stands for the name you used for this TextVectorizer class.')
        
    def text_encoder(self, text):
        if isinstance(text, list):
            return [self(t) for t in text]
        
        tks = self.tokenize(text)
        out = [self.vocab_to_idx[tk] for tk in tks]
        return out
            
    def text_decoder(self, text_ids, sep=" "):
        if all(isinstance(ids, Iterable) for ids in text_ids):
            return [self.text_decoder(ids, sep) for ids in text_ids]
            
        out = []
        for text_id in text_ids:
            out.append(self.idx_to_vocab[text_id])
            
        return f'{sep}'.join(out)
    
    def _save_json_file(self, dic, fpath):
        if exists(fpath):
            print(f"{fpath} already exists. Do you want to overwrite it?")
            print("Press [N/n] for NO, or [any other key] to overwrite.")
            confirm = input()
            if confirm in ['N', 'n']:
                return
        
        with open(fpath, 'w') as f:
            json.dump(dic, f)
            print(f"{fpath} has been successfully saved!")    
    
    def save_vocab_as_json(self, v_to_i_fpath='vocab_to_idx.json'):
        
        fmt_conv = lambda x: x + '.json' if not x.endswith('.json') else x
        v_to_i_fpath = fmt_conv(v_to_i_fpath)
        self._save_json_file(self.vocab_to_idx, v_to_i_fpath)
        
    def load_vocab_from_json(self, v_to_i_fpath='vocab_to_idx.json', msg=False):
        
        if exists(v_to_i_fpath):
            vocab_to_idx = json.load(open(v_to_i_fpath))
            self.vocab_to_idx = defaultdict(lambda: 1, vocab_to_idx)
            self.idx_to_vocab = {idx: tk for tk, idx in self.vocab_to_idx.items()}
            
            if msg:
                print(f"{v_to_i_fpath} has been successfully loaded!" \
                 + " Please call \033[1mX.vocab_to_idx\033[0m to find out more.")
                print(f"X.idx_to_vocab has been been successfully built from X.vocab_to_idx." \
                 + " Please call \033[1mX.idx_to_vocab\033[0m to find out more.")
                print('\nWhere [X] stands for the name you used for this TextVectorizer class.')
        else:   
            raise RuntimeError(f"{v_to_i_fpath} does not exists!")
    
    def __len__(self):
        return len(self.vocab_to_idx)
    
    def __call__(self, text):
        if self.vocab_to_idx:
            return self.text_encoder(text)
        raise ValueError("No vocab is built!")


def example_converter(example, text_encoder, include_seq_len):
    
    text_a, text_b, label = example
    encoded_a = text_encoder(text_a)
    encoded_b = text_encoder(text_b)
    if include_seq_len:
        len_a, len_b = len(encoded_a), len(encoded_b)
        return encoded_a, encoded_b, len_a, len_b, label
    return encoded_a, encoded_b, label


def get_trans_fn(text_encoder, include_seq_len):
    return lambda ex: example_converter(ex, text_encoder, include_seq_len)


def get_batchify_fn(include_seq_len):
    
    if include_seq_len:
        stack = [Stack(dtype="int64")] * 3
    else:
        stack = [Stack(dtype="int64")]
    
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=0),  
        Pad(axis=0, pad_val=0),  
        *stack
    ): fn(samples)
    
    return batchify_fn


def create_dataloader(dataset, 
                      trans_fn, 
                      batchify_fn, 
                      batch_size=64, 
                      shuffle=True, 
                      sampler=BatchSampler):
    
    
    if not isinstance(dataset, MapDataset):
        dataset = MapDataset(dataset)
        
    dataset.map(trans_fn)
    batch_sampler = sampler(dataset, 
                            shuffle=shuffle, 
                            batch_size=batch_size)
    
    dataloder = DataLoader(dataset, 
                           batch_sampler=batch_sampler, 
                           collate_fn=batchify_fn)
    
    return dataloder
