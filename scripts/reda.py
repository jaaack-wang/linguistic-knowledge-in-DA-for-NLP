'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: A adapted and rewriten version of Esay Data Augmentation (EDA)
for Chinese. EDA, see: https://github.com/jasonwei20/eda_nlp.
'''
from utils import readJson
from random import sample, choice, randint
from itertools import groupby
import jieba


class REDA:
    
    def __init__(self, syn_path=None):
        # path to the synonym dictionary. the dictionary must be in json format
        #  where every word is mapped to a list of synonym(s)
        if not syn_path:
            syn_path = '../data/synonyms.json'
        self._syn = readJson(syn_path)
        
    def _get_random_syn(self, word):
        if word in self._syn:
            return choice(self._syn[word])
        
    def _replaceable_idx(self, words):
        return [i for i in range(len(words)) if words[i] in self._syn]
    
    @staticmethod
    def _out(func, out_num):
        # a func to helps deduplicate the outputs
        if out_num == 1:
            return func()
        
        out = []
        for _ in range(out_num):
            out.append(func())
            
        out.sort()        
        return [o for o,_ in groupby(out)]
    
    @staticmethod
    def _out_str(words, out_str):
        # a func to return the output in a right format 
        if out_str:
            return ''.join(words)
        return words
    
    def replace_syn(self, words, rpl_num=1, out_num=1, out_str=False):
        '''Replaces given num of synonyms for randomly chosen eligible words for given times.'''
        def replace():
            words_copy = words.copy()
            rpl_idx = sample(replaceable_idx, rpl_num)
            for i in rpl_idx:
                words_copy[i] = self._get_random_syn(words[i])
            return self._out_str(words_copy, out_str)
        
        replaceable_idx = self._replaceable_idx(words)
        if len(replaceable_idx) == 0:
            return self._out_str(words, out_str)
        if len(replaceable_idx) < rpl_num:
            rpl_num = len(replaceable_idx)
        return self._out(replace, out_num)
    
    def swap_words(self, words, swap_num=1, out_num=1, out_str=False):
        '''Swaps given num of random pairs of words for given times.'''
        def swap():
            words_copy = words.copy()
            for i in range(swap_num):
                i1, i2 = sample(idx, 2)
                words_copy[i1], words_copy[i2] = words_copy[i2], words_copy[i1]
            return self._out_str(words_copy, out_str)
        
        if len(words) == 1:
            return self._out_str(words, out_str)
        if len(words) == 2 and words[0] == words[1]:
            return self._out_str(words, out_str)
        idx = list(range(len(words)))
        return self._out(swap, out_num)
            
    
    def insert_words(self, words, insert_num=1, out_num=1, out_str=False):
        '''Inserts given num of synonyms for randomly chosen eligible words for given times.'''
        def insert():
            words_copy = words.copy()
            idx_ins = sample(insertable_idx, insert_num)
            for i in idx_ins:
                words_copy.insert(i + 1, self._get_random_syn(words[i]))
            return self._out_str(words_copy, out_str)
        
        insertable_idx = self._replaceable_idx(words)
        if len(insertable_idx) == 0:
            return self._out_str(words, out_str)
        if len(insertable_idx) < insert_num:
            insertable_idx = insertable_idx * (round(insert_num / len(insertable_idx)) + 1)
        return self._out(insert, out_num)
    
    def delete_words(self, words, delete_num=1, out_num=1, out_str=False):
        '''Deletes given num of random words for given times.'''
        def delete():
            idx_del = sample(idx, delete_num)
            return self._out_str([words[i] for i in idx if i not in idx_del], out_str)
        
        if len(words) == 1:
            return self._out_str(words, out_str)
        assert len(words) > delete_num, 'The num of words to delete must be less than the num of words'
        idx = list(range(len(words)))
        return self._out(delete, out_num)
        
    def mixed_edits(self, words, max_mix=None, out_num=1, out_str=False):
        '''Mixes 2-4 of the above four operations randomly for given times.'''
        def mixed_edit():
            words_copy = words.copy()
            for edit in edits:
                words_copy = edit(words_copy)
            return self._out_str(words_copy, out_str)
        
        if max_mix:
            assert max_mix >= 2 and max_mix <= 4, 'The num of "max_mix" must be 2, 3, or 4'
            max_mix = 2
        else:
            max_mix = randint(2, 4)
        edits = sample([self.replace_syn, self.swap_words, 
                        self.insert_words, self.delete_words], max_mix)
        return self._out(mixed_edit, out_num)
    
    @staticmethod
    def tokenize(text):
        '''Chinese tokenizer.'''
        return jieba.lcut(text)    
    
    def augment_text(self, text, 
                     replace_rate=0.2, swap_rate=0.2, 
                     insert_rate=0.1, delete_rate=0.1, max_mix=None, 
                     out_num_each=2, max_out_num=None, out_str=True):
        '''Returns a list of augmented texts by synonym replacements,  
        random swap, random insertion, random deletion, and mixed operations.
        
        Args:
            - text (str or list): text to augment.
            - replace_rate (float): synonym replacement rate based on words. Defaults to 0.2.
            - swap_rate (float): random swap rate. Defaults to 0.2.
            - insert_rate (float): random insertion rate. Defaults to 0.1.
            - delete_rate (float): random deletion rate. Defaults to 0.1.
            - max_mix (int): the max num of mixed operations. If given, must be 2, 3, or 4.
            - out_num_each (int): num of outputs for each operation. Defaults to 2.
            - max_out_num (int): max num of augmented texts per augmentation. If not given, return all.
            - out_str (bool): whether to return the augmented text as str or a list of tokens.
        
        Returns:
            Augmented texts (list): either a list of string or a list of list where the nested list saves tokens.
        '''
        def _filter(item):
            '''A func to make sure that the data structure is all right as some operation might fail to augment 
            the text (e.g., too short, no synonyms etc.)'''
            if isinstance(item, str):
                return []
            if not out_str and isinstance(item[0], str):
                if len(item) == len(words):
                    for i in range(words_num):
                        if item[i] == words[i]:
                            return []
                return [item]
            return item
        
        if isinstance(text, str):
            words = self.tokenize(text)
        elif isinstance(text, list):
            words = text
        else:
            raise TypeError("The input text must be either a str or a list")
        
        words_num = len(words)
        replace_num = round(replace_rate * words_num) 
        swap_num = round(swap_rate * words_num) 
        insert_num = round(insert_rate * words_num) 
        delete_num = round(delete_rate * words_num) 
        
        out = []
        if replace_num:
            out.extend(_filter(self.replace_syn(words, replace_num, out_num_each, out_str)))
        if swap_num:
            out.extend(_filter(self.swap_words(words, swap_num, out_num_each, out_str)))
        if insert_num:
            out.extend(_filter(self.insert_words(words, insert_num, out_num_each, out_str)))
        if delete_num:
            out.extend(_filter(self.delete_words(words, delete_num, out_num_each, out_str)))
        out.extend(_filter(self.mixed_edits(words, max_mix, out_num_each, out_str)))
        
        # if the max_out_num is less than the real out num, do random sampling
        if max_out_num:
            if max_out_num < len(out):
                out = sample(out, max_out_num)
        # to deduplicate the outputs and ensure that the original text is no returned.
        words = self._out_str(words, out_str)
        out.append(words)      
        out.sort()
        out = [o for o,_ in groupby(out)]
        out.remove(words)
        return out