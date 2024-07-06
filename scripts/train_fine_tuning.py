import sys
import os

# 현재 스크립트의 디렉터리 경로를 기준으로 상대 경로를 절대 경로로 변환
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'models')
sys.path.append(models_dir)

# bert_fine_tuning 모듈에서 train_model 함수 임포트
from models.bert_fine_tuning import train_model

if __name__ == "__main__":
    train_model()
