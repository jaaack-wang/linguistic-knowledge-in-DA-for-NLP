from functools import partial
import argparse
import sys
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
# from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad

from data import create_dataloader, load_dataset
from data import convert_pointwise_example as convert_example
from model import PointwiseMatching

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--out_dir", type=str, default="./", help="The out directory for the prediction files.")
parser.add_argument("--train_set_suffix", type=str, default="", help="Train set suffix that can be used to distinguish the output test file.")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=64, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def predict(model, data_loader):
    """
    Predicts the data labels.

    Args:
        model (obj:`SemanticIndexBase`): A model to extract text embedding or calculate similarity of text pair.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """
    batch_probs = []

    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data

            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)

            batch_prob = model(
                input_ids=input_ids, token_type_ids=token_type_ids).numpy()

            batch_probs.append(batch_prob)

        batch_probs = np.concatenate(batch_probs, axis=0)

        return batch_probs


if __name__ == "__main__":
    paddle.set_device(args.device)

    # If you want to use ernie1.0 model, plesace uncomment the following code
    # pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained("ernie-1.0")
    # tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained('ernie-1.0')

    pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(
        'ernie-gram-zh')
    tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(
        'ernie-gram-zh')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
    ): [data for data in fn(samples)]

    valid_ds = load_dataset(args.input_file)
    examples, real_labels = [], []
    for vd in valid_ds:
        examples.append({'query': vd['query'], 'title': vd['title']})
        real_labels.append(vd['label'])

    # valid_ds = load_dataset(
    #     read_text_pair, data_path=args.input_file, lazy=False)

    valid_data_loader = create_dataloader(
        valid_ds,
        mode='predict',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    model = PointwiseMatching(pretrained_model)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    y_probs = predict(model, valid_data_loader)
    y_preds = np.argmax(y_probs, axis=1)

    accu = lambda x, y: sum([1 if a == b else 0 for a, b in zip(x, y)]) / len(x)
    accuracy = round(accu(real_labels, y_preds), 3)
    print("Accuracy: ", accuracy)

    if args.train_set_suffix:
        train_set_suffix = args.train_set_suffix
    else:
        train_set_suffix = ""

    if '.' in args.input_file:
        suffix = '.' + args.input_file.split('.')[-1]
        pred_file_path = f"{args.out_dir}test{train_set_suffix}_ernieGram_predictions{suffix}"
    else:
        pred_file_path = f"{args.out_dir}test{train_set_suffix}__ernieGram_predictions"

    with open(pred_file_path, 'w') as f:
        f.write("Prediction Accuracy: " + str(accuracy) + "\n")
        tmp = '\n{}\t{}\t{}\t{}'
        f.write(tmp.format('Text One', 'Text Two', 'Obervation', 'Prediction'))
        for idx, example in enumerate(examples):
            f.write(tmp.format(example['query'], example['title'], real_labels[idx], y_preds[idx]))
        f.close()
        print(f"The predictions for {args.input_file} have been saved in {pred_file_path}")
