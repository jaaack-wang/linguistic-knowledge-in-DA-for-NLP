'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: A pre-trained statistical n-gran Language Model with stupid backoff.
'''
from utils import ngramGenerator, readJson
import math
import jieba


class NgramLM:
    
    def __init__(self):
        self._unigram = readJson('../data/BaiduKnowsUnigram.json')
        self._bigram = readJson('../data/BaiduKnowsBigram.json')
        self._trigram = readJson('../data/BaiduKnowsTrigram.json')
        self._fourgram = readJson('../data/BaiduKnowsFourgram.json')
    
    def _fourGramProb(self, four_tk):
        if four_tk in self._fourgram:
            return math.log(self._fourgram[four_tk])
        
        prob = 0
        tri_tks = ngramGenerator(four_tk.split(), 3)
        for tri_tk in tri_tks:
            prob += self._triGramProb(tri_tk)
        return prob
        
    def _triGramProb(self, tri_tk):
        if tri_tk in self._trigram:
            return math.log(self._trigram[tri_tk])
        
        prob = 0
        bi_tks = ngramGenerator(tri_tk.split(), 2)
        for bi_tk in bi_tks:
            prob += self._biGramProb(bi_tk)
        return prob
    
    def _biGramProb(self, bi_tk):
        if bi_tk in self._bigram:
            return math.log(self._bigram[bi_tk])
        
        prob = 0
        for tk in bi_tk.split():
        # -3 is about = math.log(0.05). We assume that if a bigram
        # does not exist, there is only about 0.05 probility of the 
        # two words being used together
            prob += self._unigramProb(tk) - 3
        return prob
    
    def _unigramProb(self, tk):
        return math.log(self._unigram[tk]) if tk in self._unigram else self._unigram['<UNK>']
    
    def _ngramModelProb(self, tokens, n, probFunc):
        prob = 0
        for ngram in ngramGenerator(tokens, n):
            prob += probFunc(ngram)
        return prob
    
    def sentProb(self, sent, re_tk_point=5, trigram=False, bigram=False):
        assert len(sent) > 0, 'Input "sent" must not be empty!'
        if isinstance(sent, (list, tuple,)):
            tokens = jieba.lcut(''.join(sent)) if len(sent) <= re_tk_point else sent
        elif isinstance(sent, str):
            tokens = jieba.lcut(sent)
        else:
            raise TypeError('Input "sent" must be either a list/tuple or a string')
        
        tokens = ['<START>'] + tokens + ['<END>']
        
        if bigram:
            return self._ngramModelProb(tokens, 2, self._biGramProb)
        if trigram:
            return self._ngramModelProb(tokens, 3, self._triGramProb)
                
        return self._ngramModelProb(tokens, 4, self._fourGramProb)
    
    def rankSentsProbs(self, sentences, re_tk_point=5, trigram=False, bigram=False, out_str=True):
        probs = [self.sentProb(tokens, re_tk_point, trigram, bigram) for tokens in sentences]
        if not out_str and isinstance(sentences[0], list):
            return [(sent, prob) for prob, sent in sorted(zip(probs, sentences), reverse=True)]
        return [(''.join(sent), prob) for prob, sent in sorted(zip(probs, sentences), reverse=True)]
    
    def pickBestSent(self, sentences, out_num=1, re_tk_point=5, trigram=False, 
                     bigram=False, show_prob=False, out_str=True):
        assert isinstance(out_num, int), 'Input "num" must be an integer'
        assert out_num >= 1, 'Input "num" must be equal to or greater than 1'
        if show_prob:
            return [res for res in self.rankSentsProbs(sentences, re_tk_point, trigram, bigram, out_str)[:out_num]]
        return [sent for sent, _ in self.rankSentsProbs(sentences, re_tk_point, trigram, bigram, out_str)[:out_num]]
