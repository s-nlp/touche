"""class implements context word embeddings, like Elmo, Bert"""
"""The meaning of the equal word can change in different context, in different batch"""
import torch.nn as nn
from src.layers.layer_base import LayerBase
from allennlp.modules.elmo import Elmo, batch_to_ids



class LayerContextWordEmbeddings(LayerBase):
    """LayerWordEmbeddings implements word embeddings."""
    def __init__(self, word_seq_indexer, gpu, freeze_word_embeddings=False, pad_idx=0):
        super(LayerContextWordEmbeddings, self).__init__(gpu)
        print ("LayerContextWordEmbeddings init")
        self.embeddings = word_seq_indexer.emb
        self.embeddings.padding_idx = pad_idx
        self.word_seq_indexer = word_seq_indexer
        self.embeddings_dim = word_seq_indexer.embeddings_dim()
        self.output_dim = self.embeddings_dim
        self.gpu = gpu

    def is_cuda(self):
        return self.embeddings.weight.is_cuda
    
    def to_gpu(self, tensor):
        if self.gpu > -1:
            return tensor.cuda(device=self.gpu)
        else:
            return tensor.cpu()

    def forward(self, word_sequences):
        character_ids = self.to_gpu(self.word_seq_indexer.batch_to_ids(word_sequences))
        word_embeddings_feature = self.embeddings(character_ids) # shape: batch_size x max_seq_len x output_dim
        word_embeddings_feature = word_embeddings_feature['elmo_representations'][1]
        self.weight = word_embeddings_feature
        return word_embeddings_feature
