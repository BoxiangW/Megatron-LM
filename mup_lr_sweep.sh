#!/bin/bash

#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH --mem=0
#SBATCH --overcommit
#SBATCH --account=llmservice_fm_text
#SBATCH --partition=batch
#SBATCH --job-name=mup_lr_sweep

#####################################################################
# muP LR Sweep Script - Dense Small Model
# 
# This script performs learning rate sweeps on a small dense model
# using muP (Maximal Update Parameterization) principles.
# The optimal LR found here should transfer to larger models.
#####################################################################

set -e

SLURM_RUN=${SLURM_RUN:-1}

# muP Base Model Configuration
# Use a small "proxy" model for HP sweeps that transfers to larger models
MODEL_DEPTH=${MODEL_DEPTH:-4}          # Number of layers
FFN_MULT=${FFN_MULT:-4}                # FFN hidden size multiplier

# muP Sweep Configuration - sweep over hidden sizes and LRs
HIDDEN_SIZE_VALUES=${HIDDEN_SIZE_VALUES:-"256 512 1024 2048 4096"}
LR_VALUES=${LR_VALUES:-"1e-2 5e-3 1e-3 5e-4 1e-4"}
OPTIMIZER=${OPTIMIZER:-adam}           # adam, muon

# Training Configuration
TRAIN_ITERS=${TRAIN_ITERS:-500}
GBS=${GBS:-32}
SEQ_LEN=${SEQ_LEN:-1024}
WARMUP_ITERS=${WARMUP_ITERS:-50}

##### Environment Setup #####
export NCCL_IB_SL=1
export NCCL_IB_TIMEOUT=19
export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NVTE_FUSED_ATTN=1  
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export PYTHONWARNINGS=ignore
export NCCL_NVLS_ENABLE=0

export WANDB_API_KEY=d1c2f26b7ce80d80fb8f3f8bb9964e7bb30782eb # optional
export WANDB_ENTITY=nvidia # optional

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

DIR="/home/boxiangw/bxlustre/poc_muon/e2etest"
EXP_BASE_NAME="mup_sweep_${OPTIMIZER}_d${MODEL_DEPTH}"
RUN_DIR="${DIR}/experiments/${EXP_BASE_NAME}"
LOG_DIR="${RUN_DIR}/logs"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
DATACACHE_DIR="${DIR}/data-cache-mup"

export TRITON_CACHE_DIR=${DIR}/triton_cache
export TORCHINDUCTOR_CACHE_DIR=${DIR}/inductor_cache
export TRITON_HOME=$TRITON_CACHE_DIR
export PYTORCH_KERNEL_CACHE_PATH="${DIR}/torch_cache/"

mkdir -p ${LOG_DIR}
mkdir -p ${DATACACHE_DIR}
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${TRITON_CACHE_DIR}
mkdir -p ${TORCHINDUCTOR_CACHE_DIR}
mkdir -p ${PYTORCH_KERNEL_CACHE_PATH}

echo "=============================================="
echo "muP LR Sweep Configuration"
echo "=============================================="
echo "Hidden Size Values: ${HIDDEN_SIZE_VALUES}"
echo "LR Values: ${LR_VALUES}"
echo "Model Depth (layers): ${MODEL_DEPTH}"
echo "FFN Multiplier: ${FFN_MULT}"
echo "Optimizer: ${OPTIMIZER}"
echo "Train Iters: ${TRAIN_ITERS}"
echo "GBS: ${GBS}"
echo "Seq Length: ${SEQ_LEN}"
echo "Log Dir: ${LOG_DIR}"
echo "=============================================="

MEGATRON_PATH="/home/boxiangw/bxlustre/general_repro/Megatron-LM"
export PYTHONPATH=${MEGATRON_PATH}:/home/boxiangw/bxlustre/general_repro/Emerging-Optimizers
CONTAINER=/home/boxiangw/bxlustre/container/25_11_rc7.sqsh

# Function to run a single experiment with given hidden_size and LR
run_experiment() {
    local HIDDEN_SIZE=$1
    local LR=$2
    
    # Compute derived values
    local FFN_HIDDEN_SIZE=$((HIDDEN_SIZE * FFN_MULT))
    local NUM_HEADS=$((HIDDEN_SIZE / 64))  # head_dim = 64
    if [ ${NUM_HEADS} -lt 1 ]; then
        NUM_HEADS=1
    fi
    
    local EXP_NAME="${EXP_BASE_NAME}_w${HIDDEN_SIZE}_lr${LR}"
    local TB_SUBDIR="${TENSORBOARD_DIR}/w${HIDDEN_SIZE}_lr${LR}"
    
    mkdir -p ${TB_SUBDIR}
    
    # muP scaling: min_lr is typically 0.1 * lr
    local MIN_LR=$(python3 -c "print(${LR} * 0.1)")
    local WARMUP_INIT=$(python3 -c "print(${LR} * 0.01)")
    
    # muP init std scaling: 1/sqrt(width)
    local INIT_STD=$(python3 -c "import math; print(1.0 / math.sqrt(${HIDDEN_SIZE}))")
    
    echo ""
    echo "=============================================="
    echo "Running: hidden_size=${HIDDEN_SIZE}, LR=${LR}"
    echo "FFN Hidden Size: ${FFN_HIDDEN_SIZE}"
    echo "Num Heads: ${NUM_HEADS}"
    echo "Init Std (muP): ${INIT_STD}"
    echo "Min LR: ${MIN_LR}"
    echo "=============================================="
    
    # Base options for optimizer
    local base_options=""
    if [[ ${OPTIMIZER} == "muon" ]]; then
        base_options="${base_options} --use-distributed-optimizer --optimizer muon --muon-momentum 0.95"
    elif [[ ${OPTIMIZER} == "adam" ]]; then
        base_options="${base_options} --use-distributed-optimizer --overlap-param-gather --optimizer adam"
    else
        base_options="${base_options} --optimizer ${OPTIMIZER}"
    fi
    
    # Full options
    local options=" ${base_options} \
        --no-bias-swiglu-fusion \
        --no-bias-dropout-fusion \
        --no-rope-fusion \
        --mock-data \
        --train-iters ${TRAIN_ITERS} \
        --distributed-timeout-minutes 60 \
        --use-mcore-models \
        --data-cache-path ${DATACACHE_DIR} \
        --no-mmap-bin-files \
        --disable-bias-linear \
        --use-flash-attn \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 \
        --micro-batch-size 1 \
        --global-batch-size ${GBS} \
        --num-layers ${MODEL_DEPTH} \
        --hidden-size ${HIDDEN_SIZE} \
        --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
        --num-attention-heads ${NUM_HEADS} \
        --max-position-embeddings ${SEQ_LEN} \
        --position-embedding-type rope \
        --rotary-base 10000 \
        --normalization RMSNorm \
        --norm-epsilon 1e-06 \
        --swiglu \
        --untie-embeddings-and-output-weights \
        --attention-dropout 0 \
        --hidden-dropout 0 \
        --bf16 \
        --transformer-impl transformer_engine \
        --seq-length ${SEQ_LEN} \
        --clip-grad 1 \
        --weight-decay 0.1 \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-warmup-init ${WARMUP_INIT} \
        --init-method-std ${INIT_STD} \
        --lr-decay-iters ${TRAIN_ITERS} \
        --lr-warmup-iters ${WARMUP_ITERS} \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --log-interval 1 \
        --eval-iters 0 \
        --tokenizer-type NullTokenizer \
        --vocab-size 32000 \
        --overlap-grad-reduce \
        --no-create-attention-mask-in-dataloader \
        --num-workers 1 \
        --log-throughput \
        --log-progress \
        --use-mup \
        --mup-multiplier $((HIDDEN_SIZE / 256)) \
        --mup-change-init-method-std \
        --tensorboard-dir ${TB_SUBDIR} \
        --wandb-project mup-sweep \
        --wandb-exp-name ${EXP_NAME} \
        --wandb-save-dir ${RUN_DIR}/wandb"
    
    GPUS_PER_NODE=${GPUS_PER_NODE:-4}
    MASTER_ADDR=localhost
    MASTER_PORT=6000
    NUM_NODES=1
    NODE_RANK=0
    WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))
    
    DISTRIBUTED_ARGS=(
        --nproc_per_node $GPUS_PER_NODE 
        --nnodes $NUM_NODES 
        --master_addr $MASTER_ADDR 
        --master_port $MASTER_PORT
    )
    
    run_cmd="torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRON_PATH}/pretrain_gpt.py ${options}"
    echo "Command: ${run_cmd}"
    ${run_cmd} 2>&1 | tee ${LOG_DIR}/${EXP_NAME}_${DATETIME}.log
}

# Run nested sweep over hidden_size and LR
echo ""
echo "Starting muP Sweep..."
echo "Hidden Size Values: ${HIDDEN_SIZE_VALUES}"
echo "LR Values: ${LR_VALUES}"
echo ""

total_runs=0
for hs in ${HIDDEN_SIZE_VALUES}; do
    for lr in ${LR_VALUES}; do
        total_runs=$((total_runs + 1))
    done
done
echo "Total number of runs: ${total_runs}"
echo ""

current_run=0
for HIDDEN_SIZE in ${HIDDEN_SIZE_VALUES}; do
    for LR in ${LR_VALUES}; do
        current_run=$((current_run + 1))
        echo ""
        echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
        echo "Run ${current_run}/${total_runs}: hidden_size=${HIDDEN_SIZE}, lr=${LR}"
        echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" 
        echo ""
        run_experiment ${HIDDEN_SIZE} ${LR}
    done
done

echo ""
echo "=============================================="
echo "muP Sweep Complete!"
echo "Results saved to: ${LOG_DIR}"
echo "TensorBoard logs: ${TENSORBOARD_DIR}"
echo "=============================================="
echo ""
echo "To compare results, run:"
echo "  tensorboard --logdir ${TENSORBOARD_DIR}"
echo ""
echo "muP validation: If the optimal LR is consistent across"
echo "different hidden sizes, the hyperparameters should transfer."
echo ""

set +x
