These files are mostly placeholders as they are either too large to be directly put here or have copyrights protected. Inside the files is the description about how to access them. I will reiterate here:

- The `synonyms.json` is light enough to directly sit here. You can learn how it was compiled and preprocessed in my [Chinese-Synonyms repository](https://github.com/jaaack-wang/Chinese-Synonyms/tree/main/Trimmed_Synonyms). 
- BaiduKnowsNgrams, including `BaiduKnowsUnigram.json`, `BaiduKnowsBigram.json`, `BaiduKnowsTrigram.json`, and `BaiduKnowsFourgram.json` can be downloaded from [this link](https://drive.google.com/file/d/1IA7HfpYrB5XRUPSV69DVKl2PANU5NvpH/view?usp=sharing). You can learn how they are compiled by visiting my [ChineseNgrams repository](https://github.com/jaaack-wang/ChineseNgrams/tree/main/WordsBased).
- The `train.txt`, `dev.txt`, `test.txt` are LCQMC (a Large-scale Chinese Question Matching Corpus). You can request a copy of them by submitting an application form at: http://icrc.hitsz.edu.cn/info/1037/1146.htm. Alternatively, to access the full train set and dev set, you can install paddlenlp (https://github.com/PaddlePaddle/PaddleNLP) and use the following code. Please note that, the test set obtained this way are deprived of labels deliberately for holding related competitions.

```python
from paddlenlp.datasets import load_dataset
train_ds, dev_ds, test_ds = load_dataset('lcqmc', splits=['train', 'dev', 'test'])
```
- `vocab.txt` is the vocabulary as in `BaiduKnowsUnigram.json`, whcih has over 1.5 million words. Please click the file see how it should be re-compiled.