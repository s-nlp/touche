"""Vanilla recurrent network model for sequences tagging."""
import torch
import torch.nn as nn
from src.models.tagger_base import TaggerBase
from src.layers.layer_word_embeddings import LayerWordEmbeddings
from src.layers.layer_bivanilla import LayerBiVanilla
from src.layers.layer_bilstm import LayerBiLSTM
from src.layers.layer_bigru import LayerBiGRU
from src.layers.layer_context_word_embeddings import LayerContextWordEmbeddings
from src.layers.layer_context_word_embeddings_bert import LayerContextWordEmbeddingsBert

utf8stdout = open(1, 'w', encoding='utf-8', closefd=False)

class TaggerBiRNN(TaggerBase):
    """TaggerBiRNN is a Vanilla recurrent network model for sequences tagging."""
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, batch_size=1, rnn_hidden_dim=100,
                 freeze_word_embeddings=False, dropout_ratio=0.5, rnn_type='GRU', gpu=-1):
        super(TaggerBiRNN, self).__init__(word_seq_indexer, tag_seq_indexer, gpu, batch_size)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.freeze_embeddings = freeze_word_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu
        if ((not word_seq_indexer.bert) and (not word_seq_indexer.elmo)):
            self.word_embeddings_layer = LayerWordEmbeddings(word_seq_indexer, gpu, freeze_word_embeddings)
        elif (word_seq_indexer.bert):
            self.word_embeddings_layer = LayerContextWordEmbeddingsBert(word_seq_indexer, gpu, freeze_word_embeddings)
        else:
            self.word_embeddings_layer = LayerContextWordEmbeddings(word_seq_indexer, gpu, freeze_word_embeddings)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        if rnn_type == 'GRU':
            self.birnn_layer = LayerBiGRU(input_dim=self.word_embeddings_layer.output_dim,
                                          hidden_dim=rnn_hidden_dim,
                                          gpu=gpu)
        elif rnn_type == 'LSTM':
            self.birnn_layer = LayerBiLSTM(input_dim=self.word_embeddings_layer.output_dim,
                                           hidden_dim=rnn_hidden_dim,
                                           gpu=gpu)
        elif rnn_type == 'Vanilla':
            self.birnn_layer = LayerBiVanilla(input_dim=self.word_embeddings_layer.output_dim+self.char_cnn_layer.output_dim,
                                           hidden_dim=rnn_hidden_dim,
                                           gpu=gpu)
        else:
            raise ValueError('Unknown rnn_type = %s, must be either "LSTM" or "GRU"')
        # We add an additional class that corresponds to the zero-padded values not to be included to the loss function
        self.lin_layer = nn.Linear(in_features=self.birnn_layer.output_dim, out_features=class_num + 1)
        self.log_softmax_layer = nn.LogSoftmax(dim=1)
        if gpu >= 0:
            self.cuda(device=self.gpu)
        self.nll_loss = nn.NLLLoss(ignore_index=0) # "0" target values actually are zero-padded parts of sequences

    def forward(self, word_sequences):
        mask = self.get_mask_from_word_sequences(word_sequences)        
        z_word_embed = self.word_embeddings_layer(word_sequences)       
        self.z_word_embed = z_word_embed
        self.word_sequences = word_sequences
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask)
        z_rnn_out = self.apply_mask(self.lin_layer(rnn_output_h), mask) # shape: batch_size x class_num + 1 x max_seq_len
        y = self.log_softmax_layer(z_rnn_out.permute(0, 2, 1))
        return y

    def get_loss(self, word_sequences_train_batch, tag_sequences_train_batch):
        outputs_tensor_train_batch_one_hot = self.forward(word_sequences_train_batch)
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
        loss = self.nll_loss(outputs_tensor_train_batch_one_hot, targets_tensor_train_batch)
        return loss
    
    def get_grads(self):
        print (self.birnn_layer.rnn._parameters['weight_ih_l0'])
        print (self.birnn_layer.rnn._parameters['weight_ih_l0'].grad)
        print (self.birnn_layer.rnn._parameters['weight_hh_l0'])
        print (self.birnn_layer.rnn._parameters['weight_hh_l0'].grad)
        #print ("word seq")
        #print (self.word_sequences[:3], file = utf8stdout)
        #print ("word emb")
        #print (self.z_word_embed.shape)
        #torch.save([self.z_word_embed], 'fnm.pth')
        #print (self.z_word_embed[0,:,:5])
        
    def get_we_l(self):
        self.word_embeddings_layer.get_out()
        