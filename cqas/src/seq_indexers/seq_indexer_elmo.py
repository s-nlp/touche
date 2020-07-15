"""a stub that passes elmo's parameters further"""
import string
import re
from src.seq_indexers.seq_indexer_base_embeddings import SeqIndexerBaseEmbeddings
from allennlp.modules.elmo import Elmo, batch_to_ids


class SeqIndexerElmo(SeqIndexerBaseEmbeddings):
    """SeqIndexerWord converts list of lists of words as strings to list of lists of integer indices and back."""
    def __init__(self, gpu=-1, check_for_lowercase=True, embeddings_dim=0, verbose=True, options_file = '', weights_file = '', num_layers_ = 2, dropout_ = 0.1):
        SeqIndexerBaseEmbeddings.__init__(self, gpu=gpu, check_for_lowercase=check_for_lowercase, zero_digits=True,
                                          pad='<pad>', unk='<unk>', load_embeddings=True, embeddings_dim=embeddings_dim,
                                          verbose=verbose, isElmo = True)
        print ("create seq indexer elmo")
        self.no_context_base = True
        self.elmo = True
        self.options_fn = options_file
        self.weights_fn = weights_file
        self.emb = Elmo(options_file, weights_file, num_layers_, dropout=dropout_)
        self.embeddings_dim = self.emb.get_output_dim
        
    def batch_to_ids(self, batch):
        return batch_to_ids(batch)
        
    
