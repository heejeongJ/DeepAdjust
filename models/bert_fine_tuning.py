from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import os

## 파인 튜닝을 위한 모델 작성
class IMDbDataset(Dataset):
    def __init__(self, text, labels, tokenizer):
        # 텍스트를 토큰화하고, 패딩과 길이 제한을 적용하여 인코딩 진행
        self.encodings = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
        self.labels = labels
        print(f"Text length: {len(text)}")
        print(f"Labels length: {len(labels)}")
        print(f"Encodings: {self.encodings.keys()}")

    def __getitem__(self, idx):             # 주어진 인덱스의 데이터 항목을 반환
        item = {key: val[idx] for key, val in self.encodings.items()}
        # 각 토큰에 대해 동일한 레이블을 설정 (이진 분류의 경우 0 또는 1)
        # 여기서는 각 시퀀스에 대해 하나의 레이블을 가지므로, 레이블을 시퀀스 길이만큼 확장.
        item['labels'] = torch.tensor([self.labels[idx]] * self.encodings['input_ids'].shape[1])
        return item

    def __len__(self):
        return len(self.labels)


def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data', 'data')

    print(f"Current directory: {current_dir}")
    print(f"Data directory: {data_dir}")

    train_texts_path = os.path.join(data_dir, 'train_texts.txt')
    train_labels_path = os.path.join(data_dir, 'train_labels.txt')
    test_texts_path = os.path.join(data_dir, 'test_texts.txt')
    test_labels_path = os.path.join(data_dir, 'test_labels.txt')

    print(f"Train texts path: {train_texts_path}")
    print(f"Train labels path: {train_labels_path}")
    print(f"Test texts path: {test_texts_path}")
    print(f"Test labels path: {test_labels_path}")

    if not os.path.exists(train_texts_path):
        print(f"File not found: {train_texts_path}")
        return None, None, None, None
    if not os.path.exists(train_labels_path):
        print(f"File not found: {train_labels_path}")
        return None, None, None, None
    if not os.path.exists(test_texts_path):
        print(f"File not found: {test_texts_path}")
        return None, None, None, None
    if not os.path.exists(test_labels_path):
        print(f"File not found: {test_labels_path}")
        return None, None, None, None

    with open(train_texts_path, encoding='utf-8') as f:
        train_texts = f.readlines()
    with open(train_labels_path, encoding='utf-8') as f:
        train_labels = list(map(int, f.readlines()))
    with open(test_texts_path, encoding='utf-8') as f:
        test_texts = f.readlines()
    with open(test_labels_path, encoding='utf-8') as f:
        test_labels = list(map(int, f.readlines()))

    return train_texts, train_labels, test_texts, test_labels




def train_model(model_name = "bert-base-uncased", output_dir = './results', num_train_epochs = 3):          # bert의 소문자 버전 모델 사용 (대소문자 구분하지 않고 모두 소문자로 변환)
    train_texts, train_labels, test_texts, test_labels = load_data()

    # 파일이 없는 경우에 대한 처리
    if train_texts is None or train_labels is None or test_texts is None or test_labels is None:
        print("Some data files are missing. Please check the paths and ensure all files are available.")
        return

    # 토크나이저와 데이터셋을 초기화
    tokenizer = AutoTokenizer.from_pretrained(model_name)                                       # from_pertrained를 통해 사전학습된 모델을 로드함.
    train_dataset = IMDbDataset(train_texts, train_labels, tokenizer)                           # IMDbDataset 클래스 초기화 시 max_length를 추가하여 인코딩
    test_dataset = IMDbDataset(test_texts, test_labels, tokenizer)

    # 모델 초기화
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels = 2)         # Huggingface의 모델 허브에서 사전학습된 가중치(weight)와 설정 (configuration)을 불러옴.
    # 분류 태스크에 사용할 레이블의 수를 num_labels를 통해 2(긍/부정 이진 분류)로 지정.
    # AutoModelForTokenClassification를 통해 사전학습된 bert모델의 마지막 레이어를 감정 분류 태스크에 맞게 조정함. 모델이 로드되면, 파인튜닝을 통해 감정 분석에 최적화 진행.


    train_args = TrainingArguments(                                                             # 훈련 인자 설정
        output_dir=output_dir,                                                                  # 모델과 학습 상태 저장할 디렉토리 경로 지정
        num_train_epochs=num_train_epochs,                                                      # 전체 훈련 데이터셋에 대하여 학습을 반복할 횟수를 지정
        per_device_train_batch_size=8,                                                          # 훈련 시 GPU, cpu 하나당 배치 크기 설정
        per_device_eval_batch_size=8,                                                           # 평가 시 GPU, cpu 하나당 배치 크기 설정
        warmup_steps=500,                                                                       # 학습 초기에 학습률을 서서히 증가시키기 위한 워밍업 단계의 스텝 수를 지정
        weight_decay=0.01,                                                                      # 학습 중 가중치 감소(regularization)를 적용하여 과적합을 방지
        logging_dir=os.path.join(os.path.dirname(__file__), '..', 'logs', 'fine_tuning'),       # 학습 로그를 저장할 디렉토리 경로 지정
        logging_steps=10,                                                                       # 로그 기록 빈도 지정 (10 스텝마다 로그 기록)
    )

    trainer = Trainer(                      # trainer 사용하여 학습 수행 (trainer 객체 초기화)
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )


    ## 모델 학습 및 저장
    trainer.train()
    trainer.save_model(output_dir)                                                              # 학습이 완료된 모델을 지정한 디렉토리에 저장
    #trainer.save_state()                                                                       # 학습 상태 저장


if __name__ == "__main__":
    train_model()