# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np

from paddlenlp.datasets import MapDataset

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

def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)


def convert_pointwise_example(example,
                              tokenizer,
                              max_seq_length=512,
                              is_test=False):

    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids  

# def gen_pair(dataset, pool_size=100):
#     """ 
#     Generate triplet randomly based on dataset
 
#     Args:
#         dataset: A `MapDataset` or `IterDataset` or a tuple of those. 
#             Each example is composed of 2 texts: exampe["query"], example["title"]
#         pool_size: the number of example to sample negative example randomly

#     Return:
#         dataset: A `MapDataset` or `IterDataset` or a tuple of those.
#         Each example is composed of 2 texts: exampe["query"], example["pos_title"]、example["neg_title"]
#     """

    # if len(dataset) < pool_size:
    #     pool_size = len(dataset)

    # new_examples = []
    # pool = []
    # tmp_exmaples = []

    # for example in dataset:
    #     label = example["label"]

    #     # Filter negative example
    #     if label == 0:
    #         continue

    #     tmp_exmaples.append(example)
    #     pool.append(example["title"])

    #     if len(pool) >= pool_size:
    #         np.random.shuffle(pool)
    #         for idx, example in enumerate(tmp_exmaples):
    #             example["neg_title"] = pool[idx]
    #             new_examples.append(example)
    #         tmp_exmaples = []
    #         pool = []
    #     else:
    #         continue
    # return MapDataset(new_examples)
