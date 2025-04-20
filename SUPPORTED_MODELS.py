# 支持的模型及本地缓存目录
SUPPORTED_MODELS = {
    "deepseek": {
        "hf_id": "deepseek-ai/deepseek-coder-1.3b-base",
        "cache_dir": "./hf_models/deepseek"
    },
    "tinyllama": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "cache_dir": "./hf_models/tinyllama"
    },
    "deepseek-r1-1.5b":{
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "cache_dir": "./hf_models/deepseek-r1-1.5b"
    },
    "qwen-1.5b": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "cache_dir": "./hf_models/qwen-1.5b"
    },

    "flan-t5-base": {
        "hf_id": "google/flan-t5-base",
        "cache_dir": "./hf_models/flan-t5-base"
    },
    "flan-t5-small": {
        "hf_id": "google/flan-t5-small",
        "cache_dir": "./hf_models/flan-t5-small"
    },
    "phi2": {
        "hf_id": "microsoft/phi-2",
        "cache_dir": "./hf_models/phi2"
    },
    "janus-1.3b": {
        "hf_id": "deepseek-ai/Janus-1.3B",
        "cache_dir": "./hf_models/janus-1.3b"
    }
}

