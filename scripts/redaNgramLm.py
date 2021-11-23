'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: A adapted and rewriten version of Esay Data Augmentation (EDA)
for Chinese combined with a ngram Language Model. 
References:
- EDA, see: https://github.com/jasonwei20/eda_nlp.
- REDA, see: reda.py in the same folder.
- NgramLM, see: ngramLM.py in the same folder.
'''
from ngramLM import NgramLM
from reda import REDA
from itertools import groupby


class REDA_NgramLM:
    
    def __init__(self, syn_path=None):
        self._reda = REDA()
        self._lm = NgramLM()
    
    @staticmethod
    def _set_choice_num(edit_num, choice_num):
        if choice_num:
            return choice_num
        if edit_num == 1:
            return 20
        if edit_num == 2:
            return 50
        if edit_num == 3:
            return 100
        if edit_num is None:
            return 150
        return edit_num * 50
    
    def _textEdit(self, editFunc, words, rpl_num, out_num, out_str, choice_num):
        choice_num = self._set_choice_num(rpl_num, choice_num)
        out = editFunc(words, rpl_num, choice_num, out_str)
        return self._lm.pickBestSent(out, out_num=out_num, out_str=out_str)
    
    def replace_syn(self, words, rpl_num=1, out_num=1, out_str=False, choice_num=None):
        return self._textEdit(self._reda.replace_syn, words, rpl_num, out_num, out_str, choice_num)
    
    def swap_words(self, words, swap_num=1, out_num=1, out_str=False, choice_num=None):
        return self._textEdit(self._reda.swap_words, words, swap_num, out_num, out_str, choice_num)
    
    def insert_words(self, words, insert_num=1, out_num=1, out_str=False, choice_num=None):
        return self._textEdit(self._reda.insert_words, words, insert_num, out_num, out_str, choice_num)
    
    def delete_words(self, words, delete_num=1, out_num=1, out_str=False, choice_num=None):
        return self._textEdit(self._reda.delete_words, words, delete_num, out_num, out_str, choice_num)
    
    def mixed_edits(self, words, max_mix=None, out_num=1, out_str=False, choice_num=None):
        return self._textEdit(self._reda.mixed_edits, words, max_mix, out_num, out_str, choice_num)
    
    def tokenize(self, text):
        return self._reda.tokenize(text) 
    
    def augment_text(self, text, 
                     replace_rate=0.2, swap_rate=0.2, 
                     insert_rate=0.1, delete_rate=0.1, max_mix=None, 
                     out_num_each=2, out_str=True):
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
        
        # to deduplicate the outputs and ensure that the original text is no returned.
        words = self._reda._out_str(words, out_str)
        out.append(words)      
        out.sort()
        out = [o for o,_ in groupby(out)]
        out.remove(words)
        return out
