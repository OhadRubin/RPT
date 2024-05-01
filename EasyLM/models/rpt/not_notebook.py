import sys
sys.path.append('RPT')
from EasyLM.models.rpt.rpt_model_torch import RPTConfig, RPTForCausalLM, RPTLowcoderRetrieverEncodedOutput, EncodedNeighbors,RPTModel
from EasyLM.models.rpt.memory_torch import Memory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.models.rpt.rpt_serve_torch import apply_forward_upcoder, apply_forward_loglikelihood, apply_forward_lowcoder, apply_forward_augment, \
 rolling_iterator,create_forward_loglikelihood, _loglikelihood_rolling
import numpy as np
import more_itertools
import pandas as pd
from sklearn.utils import Bunch
import more_itertools

# load hf_model
hf_model = RPTForCausalLM.from_pretrained("rpt-torch-1",input_shape=(1,128))




tokenizer = RPTConfig.get_tokenizer(truncation_side='left', padding_side='left')


_forward_upcoder = apply_forward_upcoder
_forward_loglikelihood = apply_forward_loglikelihood
_forward_lowcoder = apply_forward_lowcoder
_forward_augment = apply_forward_augment

forward_loglikelihood = create_forward_loglikelihood(hf_model.config, _forward_lowcoder, _forward_upcoder, _forward_loglikelihood)
def loglikelihood_rolling(text):
    return _loglikelihood_rolling(tokenizer, hf_model, text,
                                  func=forward_loglikelihood,
                                  nearest_chunk_distance=16,
                                  return_scores=True
                                    )


with open('input.txt',"r") as f:
    text = f.read()

output = loglikelihood_rolling([text])

input_tokens = np.concatenate([x['input_tokens'] for x in rolling_iterator(tokenizer, text, input_length=1024)])
total_loglikelihood, total_is_greedy, token_count, metadata_list = output
token_count = np.array(token_count).sum()
log_likelihood = np.array(total_loglikelihood).sum()
perplexity = np.exp(-log_likelihood/token_count)
print(perplexity)
def create_str_lists(input_ids):
    queries_input_idxs = input_ids.reshape([-1,64])
    corpus_input_ids = more_itertools.sliding_window(queries_input_idxs,2)
    corpus_input_ids = map(np.concatenate, corpus_input_ids)
    corpus_input_ids = np.array(list(corpus_input_ids))
    corpus_input_ids_str = tokenizer.batch_decode(corpus_input_ids)
    queries_input_idxs_str = tokenizer.batch_decode(queries_input_idxs)
    return queries_input_idxs_str, corpus_input_ids_str

def create_dataframe(el):
    scores = el['scores']
    chunk_index = el['chunk_index']
    n_chunks_scored = scores.shape[1]
    n_seg_chunks = scores.shape[0]
    dataframe = dict(
        key_index=np.broadcast_to(np.arange(n_chunks_scored)[None,:],(n_seg_chunks,n_chunks_scored)).reshape(-1),
        query_index=np.broadcast_to(chunk_index[:,None],(n_seg_chunks,n_chunks_scored)).reshape(-1),
        score=scores.reshape(-1)
        )
    return pd.DataFrame(dataframe)
queries_input_idxs_str, corpus_input_ids_str = create_str_lists(input_tokens)


df_list = [create_dataframe(el) for el in metadata_list[2:]] #first 32 chunks, we do not allow to retrieve, so these predictions will be empty.
df = pd.concat(df_list)

def top_k(x, k):
    return x.sort_values("score",ascending=False).head(k).reset_index() # note that topk=0 means the highest score, since ascending=False.
top_df = df.groupby("query_index").apply(lambda x: top_k(x, 2)).drop(columns=["query_index","index"]).reset_index().rename(columns={"level_1":"topk"})




element = Bunch(**top_df.sample(1).to_dict("records")[0])
print(element)
print(queries_input_idxs_str[element.query_index]) # query chunk
print("----")
print(queries_input_idxs_str[element.query_index+1]) # target chunk
print("----")
print(corpus_input_ids_str[element.key_index]) # neighbor chunks + continuation