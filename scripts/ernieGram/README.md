## Model

`model.py` contains a pointwise matching network whose embeddings are initialized by the pre-trained ERNIE-Gram model. 

More about ERNIE-Gram and ERNIE:
- [ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding](https://arxiv.org/abs/2010.12148)
- [ERNIE 2.0: A Continual Pre-training Framework for Language Understanding](https://arxiv.org/abs/1907.12412)
- [ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129)


## Training

On GPU:

```cmd
python -u -m paddle.distributed.launch --gpus "0" train_pointwise.py \
        --device gpu \
		--train_set_path './train.txt' \
		--dev_set_path './dev.txt'
		--epochs 3 \
        --save_dir ./checkpoints \
        --batch_size 64 \
        --learning_rate 2E-5
```

Meaning of the parameters:

- train_set_path: path to the train set, defaults to './train.txt'.
- dev_set_path: path to the dev set, defaults to './dev.txt'.
- device: gpu or cpu, defaults to cpu.
- lr: learning rate, defaults to 5e-4. 
- batch_size: defaults to 64. Usually the batch size is multiples of 8.
- epochs: number of epochs to loop through (one epoch meaning the training goes through the entire dataset once), defaults to 10.
- save_dir: directory to save the trained model's weights.

Other parameters available:
- max_seq_length: The maximum total input sequence length after tokenization, defaults to 128. 
- weight_decay: how much weight decay to apply, defaults to 0.0. 
- eval_step: Step interval for evaluation on dev set, defaults to 100.
- save_step: Step interval for trained parameters in the saving checkpoint dir.
- warmup_proportion: Linear warmup proption over the training process, defaults to 0.0. 

If you do not have any access to GPU, you can go to [Baidu's AI Studio](https://aistudio.baidu.com/aistudio/index) to access 8 hours free GPU every day. 


## Predicting

On GPU:

```cmd
!python -u -m paddle.distributed.launch --gpus "0" \
        predict_pointwise.py \
        --out_dir './test_preds/' \
        --train_set_suffix {train_set_suffix} \
        --device gpu \
        --params_path "./checkpoints/model_17029/model_state.pdparams"\
        --batch_size 64 \
        --input_file 'test.txt'
```

This will print the accuracy of the model on the test set and output a test prediction file named after the test file, the network model. There is a convinient variable you can call to specify a suffix for the test set, which I named as `train_set_suffix` for the convenience of conduting this research where I have a bunch of train set files to take care of.



