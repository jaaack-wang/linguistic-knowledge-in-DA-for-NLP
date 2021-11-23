import numpy as np
from paddlenlp.datasets import MapDataset

def convert_example(example, tokenizer, is_test=False):
    query, title = example["query"], example["title"]
    query_ids = np.array(tokenizer.encode(query), dtype="int64")
    query_seq_len = np.array(len(query_ids), dtype="int64")
    title_ids = np.array(tokenizer.encode(title), dtype="int64")
    title_seq_len = np.array(len(title_ids), dtype="int64")

    if not is_test:
        label = np.array(example["label"], dtype="int64")
        return query_ids, title_ids, query_seq_len, title_seq_len, label
    else:
        return query_ids, title_ids, query_seq_len, title_seq_len


def load_dataset(datafiles):
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header
            for line in fp.readlines():
                items = line.strip('\n').split('\t')
                sens1 = items[0].split('\002')[0]
                sens2 = items[1].split('\002')[0]
                labels = int(items[-1].split('\002')[0])
                yield {'query': sens1, 'title': sens2, 'label': labels}

    if isinstance(datafiles, str):
        return MapDataset(list(read(datafiles)))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):
        return [MapDataset(list(read(datafile))) for datafile in datafiles]


def preprocess_prediction_data(data, tokenizer):
    examples, labels = [], []
    for example in data:
        query_ids = tokenizer.encode(example["query"])
        title_ids = tokenizer.encode(example["title"])
        examples.append([query_ids, title_ids, len(query_ids), len(title_ids)])
        labels.append(example["label"])
    return examples, labels
