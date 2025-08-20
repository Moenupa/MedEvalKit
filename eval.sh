#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=meng.wang@cair-cas.org.hk
#SBATCH --mem-per-gpu=64G
#SBATCH -c 24
#SBATCH -p a100
#SBATCH -t 30-00:00:00
#SBATCH -o logs/job/%j.out
#SBATCH -e logs/job/%j.err
#SBATCH -J job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# MMMU-Medical-test,MMMU-Medical-val,PMC_VQA,MedQA_USMLE,MedMCQA,PubMedQA,OmniMedVQA,Medbullets_op4,Medbullets_op5,MedXpertQA-Text,MedXpertQA-MM,SuperGPQA,HealthBench,IU_XRAY,CheXpert_Plus,MIMIC_CXR,CMB,CMExam,CMMLU,MedQA_MCMLE,VQA_RAD,SLAKE,PATH_VQA,MedFrameQA,Radrestruct
EVAL_DATASETS="MMMU-Medical-val,PMC_VQA,MIMIC_CXR,VQA_RAD,SLAKE,PATH_VQA" 
DATASETS_PATH="/public/datasets/lmm-eval"
OUTPUT_PATH="eval_results/{}"
# TestModel,Qwen2-VL,Qwen2.5-VL,BiMediX2,LLava_Med,Huatuo,InternVL,Llama-3.2,LLava,Janus,HealthGPT,BiomedGPT,Vllm_Text,MedGemma,Med_Flamingo,MedDr
MODEL_NAME="Qwen2.5-VL"
MODEL_PATH="/public/models/Qwen2.5-VL-7B-Instruct"

#vllm setting
CUDA_VISIBLE_DEVICES="0"
TENSOR_PARALLEL_SIZE="1"
USE_VLLM="False"

#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1


# Eval LLM setting
MAX_NEW_TOKENS=8192
MAX_IMAGE_NUM=6
TEMPERATURE=0
TOP_P=0.0001
REPETITION_PENALTY=1

# LLM judge setting
USE_LLM_JUDGE="True"
# gpt api model name
GPT_MODEL="gpt-4.1-2025-04-14"
JUDGE_MODEL_TYPE="openai"  # openai or gemini or deepseek or claude

# load via .env file
# API_KEY=""
# BASE_URL=""
# if USE_LLM_JUDGE is "True", then API_KEY and BASE_URL must be set
if [ -f ".env" ]; then
    echo "loading from .env"
    source .env
    # Check if the required environment variables are set
    if [ -z "$API_KEY" ]; then
        echo "Warning: \`API_KE' not in ./.env."
    fi

    if [ -z "$BASE_URL" ]; then
        echo "Warning: \`BASE_URL' not in ./.env."
    fi
else
    echo ".env file not found. Please create a .env file with the required variables."
fi

echo "Running with GPU $CUDA_VISIBLE_DEVICES"
uv run python -c "import torch; print('cuda:', torch.cuda.is_available())"

# pass hyperparameters and run python sccript
uv run eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed $SEED \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --use_vllm "$USE_VLLM" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --reasoning "$REASONING" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_model_type "$JUDGE_MODEL_TYPE" \
    --judge_model "$GPT_MODEL" \
    --api_key "$API_KEY" \
    --base_url "$BASE_URL" \
    --test_times "$TEST_TIMES" \
