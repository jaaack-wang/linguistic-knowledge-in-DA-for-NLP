## Models 

`model.py` contains four nerual models, often referred to as SimNet (Similarity Networks), for short text macthing using Baidu's `paddlepaddle` framework: 

- BOW: Bag of Words
- CNN: Convolutional Neural Network
- LSTM: Long Short-Term Memory
- GRU: Gated Recurrent Units 


This basic structure is simple: an input layer that takes a pari of texts, a representation/embedding layer that encode the texts, a matching layer that leads to a score that finally goes to the output layer, a softmax activation function.


## Training

Training is easy. You can train a model on CPU and GPU. 

You first need to have a dictionary formatted as in `vocab.txt`, a train set and dev set that has a pair of texts along with a label seprated by tab ("\t") each line. 

Then, on CPU:

```cmd
python train.py --vocab_path='./vocab.txt' \
   --train_set_path='./train.txt' \
   --dev_set_path ='./dev.txt' \
   --device=cpu \
   --network=lstm \
   --lr=5e-4 \
   --batch_size=64 \
   --epochs=3 \
   --save_dir='./checkpoints'
```

on GPU:

```cmd
python -m paddle.distributed.launch --gpus "0" train.py --vocab_path='./vocab.txt' \
   --train_set_path='./train.txt' \
   --dev_set_path ='./dev.txt' \
   --device=cpu \
   --network=lstm \
   --lr=5e-4 \
   --batch_size=64 \
   --epochs=3 \
   --save_dir='./checkpoints'
```

Meaning of the parameters:

- train_set_path: path to the train set, defaults to './train.txt'.
- dev_set_path: path to the dev set, defaults to './dev.txt'.
- device: gpu or cpu, defaults to cpu.
- network: bow, cnn, lstm, or gru, ALL in lower case, defaults to lstm.
- lr: learning rate, defaults to 5e-4. 
- batch_size: defaults to 64. Usually the batch size is multiples of 8.
- epochs: number of epochs to loop through (one epoch meaning the training goes through the entire dataset once), defaults to 10.
- save_dir: directory to save the trained model's weights.



If you do not have any access to GPU, you can go to [Baidu's AI Studio](https://aistudio.baidu.com/aistudio/index) to access 8 hours free GPU every day. 


## Predicting

On CPU:

```cmd
python predict.py --vocab_path='./vocab.txt' \
   --test_set_path= './test.txt' \
   --out_dir= './' \
   --device=cpu \
   --network=lstm \
   --params_path=checkpoints/final.pdparams
```

On GPU:

```cmd
CUDA_VISIBLE_DEVICES=0 python predict.py --vocab_path='./simnet_vocab.txt' \
   --test_set_path= './test.txt' \
   --out_dir= './' \
   --device=gpu \
   --network=lstm \
   --params_path='./checkpoints/final.pdparams'
```

This will print the accuracy of the model on the test set and output a test prediction file named after the test file, the network model. There is a convinient variable you can call to specify a suffix for the test set, which I named as `train_set_suffix` for the convenience of conduting this research where I have a bunch of train set files to take care of.



