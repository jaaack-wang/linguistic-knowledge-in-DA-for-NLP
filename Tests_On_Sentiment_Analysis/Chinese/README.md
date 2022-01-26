## Description

This folder stores code, data, and experimental results for additional tests on Chinese sentiment analysis when four supervised classification models (`ERNIE-Gram` was excluded here because it was expensive to train and may not reveal anything interesting) are trained on three types of train sets of varying sizes: un-augmented train sets, train sets augmented by the `reda` program, and train sets augmented by the `reda` program combined with a `ngram` model. 

The [Chinese Sentiment Corpus](https://ccc.inaoep.mx/~villasen/bib/An%20empirical%20study%20of%20sentiment%20analysis%20for%20chinese%20documents.pdf) is used here for text augmentation and model training. My [text-classification-explained](https://github.com/jaaack-wang/text-classification-explained) repository explains how the dataset can be obtained, which also provides the four models for training Chinese sentiment classifier as in the `paddle_models` folder.

I reused the code for my preprint to augment Chinese. The whole text augmentation process is recorded in the `Aug_Texts` folder. 


## Basic statistics about the dataset

- For the train, dev, and test set.

| Dataset | Total | Positive| Negative | 
| :---: | :---: | :---: | :---: |
| Train | 9,600 | 4,798 | 4,802 | 
| Dev | 1,200 | 590 | 610 | 
| Test | 1,200 | 602 | 598 | 


- For the augmented train sets.

| Base | 0.5k | 1k | 3k| 6k| full (9k)|
| :---: | :---: | :---: | :---: | :---: | :---: |
| + REDA | 3,997 | 7,992 | 23,991 | 47,994 | 57,576 |
| + REDA + Ngram | 3,991 | 7,980 | 23,970 | 47,953 | 57,520 |


## Results

I only trained four classifcation models (BoW, CNN, LSTM-RNN, GRU-RNN), the details of which can be seen in the `Training.ipynb` file. As the data size is small, so you can re-run the script on your own computer for few hours.

Results Coming soon.

## Findings

Forthcoming.