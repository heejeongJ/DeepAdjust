from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

## 파인 튜닝을 위한 모델 작성
class IMDbDataset(Dataset):
    def __init__(self, text, labels, tokenizer):
        self.encodings = tokenizer(text, truncation = True, padding = True)
        self.labels = labels

    def __getitem__(self, idx):             # 주어진 인덱스의 데이터 항목을 반환
        item = {key : torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def load_datas():
    with open('data/train_texts.txt') as f:
        train_texts = f.readlines()
    with open('data/train_labels.txt') as f:
        train_labels = list(map(int, f.readlines()))

    with open('data/test_texts.txt') as f:
        test_texts = f.readlines()
    with open('data/test_labels.txt') as f:
        test_labels = list(map(int, f.readlines()))

    return train_texts, train_labels, test_texts, test_labels




def train_model(model_name = "bert-base-uncased", output_dir = './results', num_train_epochs = 3):          # bert의 소문자 버전 모델 사용 (대소문자 구분하지 않고 모두 소문자로 변환)
    train_texts, train_labels, test_texts, test_labels = load_datas()

    tokenizer = AutoTokenizer.from_pretrained(model_name)                                                   # from_pertrained를 통해 사전학습된 모델을 로드함.
    train_dataset = IMDbDataset(train_texts, train_labels, tokenizer)
    test_dataset = IMDbDataset(test_texts, test_labels, tokenizer)

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels = 2)                     # Huggingface의 모델 허브에서 사전학습된 가중치(weight)와 설정 (configuration)을 불러옴.
    # 분류 태스크에 사용할 레이블의 수를 num_labels를 통해 2(긍/부정 이진 분류)로 지정.
    # AutoModelForTokenClassification를 통해 사전학습된 bert모델의 마지막 레이어를 감정 분류 태스크에 맞게 조정함. 모델이 로드되면, 파인튜닝을 통해 감정 분석에 최적화 진행.


    train_args = TrainingArguments(
        output_dir=output_dir,              # 모델과 학습 상태 저장할 디렉토리 경로 지정
        num_train_epochs=num_train_epochs,  # 전체 훈련 데이터셋에 대하여 학습을 반복할 횟수를 지정
        per_device_train_batch_size=8,      # 훈련 시 GPU, cpu 하나당 배치 크기 설정
        per_device_eval_batch_size=8,       # 평가 시 GPU, cpu 하나당 배치 크기 설정
        warmup_steps=500,                   # 학습 초기에 학습률을 서서히 증가시키기 위한 워밍업 단계의 스텝 수를 지정
        weight_decay=0.01,                  # 학습 중 가중치 감소(regularization)를 적용하여 과적합을 방지
        logging_dir='logs/fine_tuning',     # 학습 로그를 저장할 디렉토리 경로 지정
        logging_steps=10,                   # 로그 기록 빈도 지정 (10 스텝마다 로그 기록)
    )

    trainer = Trainer(                      # trainer 사용하여 학습 수행
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )


    ## 모델 학습 및 저장
    trainer.train()
    trainer.save_model(output_dir)      # 학습이 완료된 모델을 지정한 디렉토리에 저장
    trainer.save_state()                # 학습 상태 저장


if __name__ == "__main__":
    train_model()