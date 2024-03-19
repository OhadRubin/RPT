

# GENERATE=1
# if [ -z "$GENERATE" ]; then
PRE_COMPILE='none'
# PRE_COMPILE='generate'
SEQ_LENGTH=1024
INPUT_LENGTH=1024

# OVERRIDE="n_windows,use_cca_norm2"
STRIDE=1024
# OVERRIDE="n_windows,stride,cca_freq"
OVERRIDE="n_windows,stride"


python3 -m EasyLM.models.rpt.rpt_serve \
    --add_outputs=True \
    --load_rpt_config="pickle::https://huggingface.co/iohadrubin/rpt-1b/raw/main/config.json" \
    --load_checkpoint="flax_params::https://huggingface.co/iohadrubin/rpt-1b/blob/main/flax_model.msgpack" \
    --mesh_dim='-1,1,1' \
    --dtype='fp32' \
    --gin_param="RPTConfig.cca_freq=0" \
    --gin_param="RPTConfig.n_windows=1" \
    --gin_param="RPTConfig.stride=$STRIDE" \
    --input_length=$INPUT_LENGTH \
    --seq_length=$SEQ_LENGTH \
    --lm_server.batch_size=1 \
    --lm_server.port=35009 \
    --override_list=$OVERRIDE \
    --top_k 50 \
    --top_p 0.9 \
    --num_neighbors 2 \
    --single_model=False \
    --lm_server.pre_compile=$PRE_COMPILE
    # --dense_mem=False \
    # --max_new_tokens=128 \
    # --override_list="stride,n_windows,cca_freq" \
    # --override_list="cca_freq,n_windows" \
    # --override_list="stride,cca_freq" \
