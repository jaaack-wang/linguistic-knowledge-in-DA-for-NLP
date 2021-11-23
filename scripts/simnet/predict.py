from functools import partial
import argparse

import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab

from model import SimNet
from utils import load_dataset, preprocess_prediction_data

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--batch_size", type=int, default=64, help="Total examples' number of a batch for training.")
parser.add_argument("--vocab_path", type=str, default="./simnet_vocab.txt", help="The path to vocabulary.")
parser.add_argument("--test_set_path", type=str, default="./test.txt", help="The path to the test set.")
parser.add_argument("--out_dir", type=str, default="./", help="The out directory for the prediction files.")
parser.add_argument("--train_set_suffix", type=str, default="", help="Train set suffix that can be used to distinguish the output test file.")
parser.add_argument('--network', type=str, default="lstm", help="Which network you would like to choose bow, cnn, lstm or gru ?")
parser.add_argument("--params_path", type=str, default='./checkpoints/final.pdparams', help="The path of model parameter to be loaded.")
args = parser.parse_args()
# yapf: enable


def predict(model, data, batch_size=1, pad_token_id=0):
    """
    Predicts the data labels.

    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).
        batch_size(obj:`int`, defaults to 1): The number of batch.
        pad_token_id(obj:`int`, optional, defaults to 0): The pad token index.

    Returns:
        results(obj:`dict`): All the predictions labels.
    """

    # Seperates data into some batches.
    batches = [
        data[idx:idx + batch_size] for idx in range(0, len(data), batch_size)
    ]

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=pad_token_id),  # query_ids
        Pad(axis=0, pad_val=pad_token_id),  # title_ids
        Stack(dtype="int64"),  # query_seq_lens
        Stack(dtype="int64"),  # title_seq_lens
    ): [data for data in fn(samples)]

    results = []
    model.eval()
    for batch in batches:
        query_ids, title_ids, query_seq_lens, title_seq_lens = batchify_fn(
            batch)
        query_ids = paddle.to_tensor(query_ids)
        title_ids = paddle.to_tensor(title_ids)
        query_seq_lens = paddle.to_tensor(query_seq_lens)
        title_seq_lens = paddle.to_tensor(title_seq_lens)
        logits = model(query_ids, title_ids, query_seq_lens, title_seq_lens)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        labels = idx.tolist()
        # labels = idx
        # # labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)
    # Loads vocab.
    vocab = Vocab.load_vocabulary(
        args.vocab_path, unk_token='[UNK]', pad_token='[PAD]')
    tokenizer = JiebaTokenizer(vocab)

    # Constructs the newtork.
    model = SimNet(
        network=args.network, vocab_size=len(vocab), num_classes=2)

    # Loads model parameters.
    state_dict = paddle.load(args.params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % args.params_path)

    # Firstly pre-processing prediction data  and then do predict.
    data = load_dataset(args.test_set_path)

    examples, real_labels = preprocess_prediction_data(data, tokenizer)
    predictions = predict(
        model,
        examples,
        batch_size=args.batch_size,
        pad_token_id=vocab.token_to_idx.get('[PAD]', 0))

    accu = lambda x, y: sum([1 if a == b else 0 for a, b in zip(x, y)]) / len(x)
    accuracy = round(accu(real_labels, predictions), 3)
    print("Accuracy: ", accuracy)
    
    if args.train_set_suffix:
        train_set_suffix = args.train_set_suffix
    else:
        train_set_suffix = ""

    if '.' in args.test_set_path:
        suffix = '.' + args.test_set_path.split('.')[-1]
        pred_file_path = f"{args.out_dir}test{train_set_suffix}_{args.network}_predictions{suffix}"
    else:
        pred_file_path = f"{args.out_dir}test{train_set_suffix}_{args.network}_predictions"

    with open(pred_file_path, 'w') as f:
        f.write("Prediction Accuracy: " + str(accuracy) + "\n")
        tmp = '\n{}\t{}\t{}\t{}'
        f.write(tmp.format('Text One', 'Text Two', 'Obervation', 'Prediction'))
        for idx, example in enumerate(data):
            f.write(tmp.format(example['query'], example['title'], example['label'], predictions[idx]))
        f.close()
        print(f"The predictions for {args.test_set_path} have been saved in {pred_file_path}")
        