# 모든 가중치 파일의 캐시 디렉토리 설정
import os

def set_weights_dir(config: str):
    """
    절대 경로 넣기
    """
    
    weights_dir = config.get('weights_dir')
    os.makedirs(weights_dir, exist_ok=True)

    # HuggingFace 관련 모든 캐시 경로 설정
    os.environ['TRANSFORMERS_CACHE'] = weights_dir
    os.environ['HF_HOME'] = weights_dir
    os.environ['HF_DATASETS_CACHE'] = os.path.join(weights_dir, 'datasets')
    os.environ['HUGGINGFACE_HUB_CACHE'] = weights_dir
    os.environ['TORCH_HOME'] = os.path.join(weights_dir, 'torch')

    return weights_dir