

import numpy as np
import jax
import math
import numpy as np
from multiprocessing import Pool, cpu_count

import pandas as pd

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


class Memory:
  def __init__(self,
               chunk_size,
               num_neighbors,
               nearest_chunk_distance,
               max_size=None,
               return_scores=False,
               is_dense=True,
               oracle=False) -> None:
    self.chunk_size = chunk_size
    self.num_neighbors = num_neighbors
    self.values_memory = None
    self.keys_memory = None
    self.bm25 = None
    self.is_dense = is_dense
    self.max_size = max_size
    self.score_matrix = None
    self.nearest_chunk_distance = nearest_chunk_distance
    self.return_scores = return_scores
    self.oracle = oracle
    self.targets = []

  def set_scores(self, nei_scores, chunk_info):
    self.score_matrix = reshape_scores(nei_scores, chunk_info)
  
  def add(self, input_tokens, append=True, **kwargs):
    # converts the elements in kwargs to np.float32
    kwargs = jax.tree_map(lambda x: jax.device_get(x).astype(np.float32), kwargs)
    return self._add(input_tokens, append=append, **kwargs)

  def _add(self, input_tokens, encoded_hidden_states, key_chunks=None, query_chunks=None, append=True):
    ret_metrics = None
    num_neighbors = self.num_neighbors

    # TODO: What is this encoded hidden state? 3D vector (curr_chunks = 16?, _? seems to be the window size 64, embedding_dim makes sense)
    curr_chunks, _, emb_dim = encoded_hidden_states.shape
    # reshape input token ta column vector
    input_tokens = list(input_tokens.reshape([-1, self.chunk_size]))
    
    metadata = {}
    
    if self.memory_size() > (self.nearest_chunk_distance+self.num_neighbors):
      chunk_scores = self.get_chunk_scores(query_chunks, input_tokens)
      neighbor_hidden_states, ret_metrics =  self.get_nei(chunk_scores, metadata)
      neighbor_mask = np.ones((curr_chunks,num_neighbors,2*self.chunk_size), dtype=bool)
    else:
      # a (curr_chunks = 16, num_neighbors = 2, 2*chunk_size=128) array of False
      neighbor_mask = np.zeros((curr_chunks,num_neighbors,2*self.chunk_size), dtype=bool)
      # a (curr_chunks = 16, num_neighbors = 2, 2*chunk_size=128, embedding_dim=2048) of 0
      neighbor_hidden_states = np.zeros((curr_chunks, num_neighbors, 2*self.chunk_size, emb_dim), dtype=np.float32)
    if append:
      self.targets.extend(input_tokens)
      if self.memory_size()>0: # add to memory,
        self.values_memory = np.concatenate([self.values_memory, encoded_hidden_states], axis=-3)
        if self.is_dense:
          self.keys_memory = np.concatenate([self.keys_memory, key_chunks], axis=-2)
        else:
          self.bm25.add_documents(input_tokens)
      else: # first time
        self.values_memory = encoded_hidden_states
        if self.is_dense:
          self.keys_memory = key_chunks
        else:
          self.bm25 = BM25(input_tokens)
        self.clip_memory()
    if not self.is_dense:
      neighbor_hidden_states, neighbor_mask = np.squeeze(neighbor_hidden_states), np.squeeze(neighbor_mask)
    return neighbor_hidden_states, neighbor_mask, metadata, ret_metrics
  
  def get_chunk_scores(self, query_chunks, input_tokens):
      if self.is_dense:
        mem = self.keys_memory[:-(self.nearest_chunk_distance+1),:]
        chunk_scores = np.einsum('nd,...bd->...bn',mem, query_chunks)/np.sqrt(mem.shape[-1])
      else:
        chunk_scores = np.stack([self.bm25.get_scores(q)[:-(self.nearest_chunk_distance+1)] for q in input_tokens])
        chunk_scores = np.expand_dims(chunk_scores,0)
      return chunk_scores
    
  def memory_size(self):
    if self.values_memory is not None:
      return self.values_memory.shape[-3]
    else:
      return 0
    
  def clip_memory(self):
      if self.max_size is not None:
        self.values_memory = self.values_memory[-self.max_size:,:,:]
        if self.is_dense:
          self.keys_memory = self.keys_memory[-self.max_size:,:]

 
  def get_nei(self,
              chunk_scores,
              metadata,
              ):
      ret_metrics = None
      chunks_seen = self.values_memory.shape[-3]
      
      curr_chunks = chunk_scores.shape[-2]
      mem_size = chunk_scores.shape[-1]
      chunk_idxs = np.arange(chunks_seen,chunks_seen+curr_chunks)
        
      if self.score_matrix is not None:
        oracle_scores = self.score_matrix[chunks_seen:chunks_seen+curr_chunks,:mem_size]
        oracle_scores = np.expand_dims(oracle_scores,0)
        oracle_scores = np.pad(oracle_scores,((0,0),(0,chunk_scores.shape[1]-oracle_scores.shape[1]),(0,0)),'constant',constant_values=-np.inf)
        
      if self.oracle:
        self.score_matrix is not None
        top_idxs = (-oracle_scores).argsort(axis=-1)[...,:self.num_neighbors]
        retrieved_scores = np.take_along_axis(oracle_scores,top_idxs,axis=-1)
      else:
        top_idxs = (-chunk_scores).argsort(axis=-1)[...,:self.num_neighbors]
        retrieved_scores = np.take_along_axis(chunk_scores,top_idxs,axis=-1)
      
      if self.score_matrix is not None:
        ret_metrics = pd.DataFrame()
        if np.isfinite(oracle_scores).any().item():
          ret_metrics = single_calc_metrics(chunk_scores,oracle_scores,chunk_idxs)
        if not self.oracle:
          overlap_metrics = single_get_overlap(chunk_idxs, self.targets, top_idxs)
          ret_metrics = pd.concat([ret_metrics,overlap_metrics])
          
      top_rep = self.values_memory[top_idxs]
      next_top_rep = self.values_memory[top_idxs+1]
      retrieved = np.concatenate([top_rep,next_top_rep],axis=-2)
      
      if self.return_scores:
        metadata['chunk_index'] = chunk_idxs
        metadata['scores'] = chunk_scores
        metadata["top_idxs"] = top_idxs
        metadata["top_scores"] = retrieved_scores
      return retrieved,ret_metrics
    
  
def reshape_scores(nei_scores, chunk_info):
    chunk_info = chunk_info.reshape(-1, 2)
    chunk_id,candidate_idx = chunk_info.T
    orig_matrix =  np.full(fill_value=-np.inf,shape=(chunk_id[-1]+2,chunk_id[-1]+2))
    orig_matrix[chunk_id,candidate_idx] = nei_scores
    score_matrix = orig_matrix[2:]  - np.diag(orig_matrix[2:,:])[:,None]
    score_matrix = np.r_[orig_matrix[:2] ,score_matrix]
    return score_matrix






class BM25:
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        self.nd = self._initialize(corpus)
        self._calc_idf()

    def add_document(self, document, recalculate_idf=True):
        if self.tokenizer:
            document = self.tokenizer(document)

        self.doc_len.append(len(document))
        self.corpus_size += 1
        self.avgdl = sum(self.doc_len) / self.corpus_size

        frequencies = {}
        for word in document:
            if word not in frequencies:
                frequencies[word] = 0
            frequencies[word] += 1
        self.doc_freqs.append(frequencies)

        for word, freq in frequencies.items():
            if word in self.nd:
                self.nd[word] += 1
            else:
                self.nd[word] = 1

        if recalculate_idf:
            self._calc_idf()


    def add_documents(self, documents):
        for document in documents:
            self.add_document(document, recalculate_idf=False)
        self._calc_idf()

    # Rest of the code remains the same

    def get_top_n(self, query, documents, n=5):

        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

    def _calc_idf(self):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in self.nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q,0)) for doc in self.doc_freqs])
            score += (self.idf.get(q,0)) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()
    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus