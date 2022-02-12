'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: Simple BoW model for text classification using paddle. 
'''
import paddle 
import paddle.nn as nn

class BoW(nn.Layer):

    def __init__(self, 
                vocab_size, 
                output_dim,
                embedding_dim=100,
                padding_idx=0,  
                hidden_dim=50, 
                activation=nn.ReLU()):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx)
        
        self.dense = nn.Linear(embedding_dim, hidden_dim)
        self.activation = activation
        self.dense_out = nn.Linear(hidden_dim, output_dim)

    def encoder(self, embd):
        # summing up the embedding text_embds
        # shape: embd (input), (batch_size, text_seq_len, embedding_dim)
        # embd (output), (batch_size, embedding_dim)
        return embd.sum(axis=1)

    def forward(self, text_ids):
        # shape: text_ids, (batch_size, text_seq_len) 
        # --> text_embd, (batch_size, text_seq_len, embedding_dim) 
        text_embd = self.embedding(text_ids)

        # shape: encoded, (batch_size, embedding_dim)
        encoded = self.encoder(text_embd)

        # go through a dense layer before output
        # shape: hidden_out, (batch_size, hidden_dim)
        hidden_out = self.activation(self.dense(encoded))

        # shape: out_logits, (batch_size, output_dim)
        # note that, since we will use cross entropy as the loss func, 
        # we will later use softmax to compute the loss
        out_logits = self.dense_out(hidden_out)
        return out_logits
