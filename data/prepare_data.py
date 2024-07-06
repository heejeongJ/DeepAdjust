from datasets import load_dataset                               # datasets 라이브러리 : 다양한 데이터셋을 쉽게 다운로드하고 사용할 수 있는 라이브러리
import os

# datasets 라이브러리에서 데이터셋을 불러오면 데이터셋은 DatasetDict 객체로 로드 됨. 해당 객체는 여러 Dataset 객체를 포함하는 딕셔너리로 구성됨.
## 데이터 준비
def prepare_data():
    dataset = load_dataset("imdb")                              # imdb는 영화 리뷰와 해당 리뷰의 감정(긍정/부정) 레이블을 포함함.
    train_texts = dataset["train"]["text"][:1000]
    train_labels = dataset["train"]["label"][:1000]
    test_texts = dataset["test"]["text"][:1000]
    test_labels = dataset["test"]["label"][:1000]



    # datasets를 텍스트 파일로 저장 (디렉토리 존재하지 않을 경우 생성)
    os.makedirs('data', exist_ok=True)

    with open(os.path.join('data', 'train_texts.text'), 'w') as f:              # 데이터셋에서 텍스트와 레이블을 추출하여 텍스트 파일로 저장, 각 데이터셋의 텍스트와 레이블을 별도의 파일로 저장하여 이후에 쉽게 접근할 수 있도록 함
        for item in train_texts:                                                # train_datasets text file로 저장
            f.write("%s\n" % item)

    with open(os.path.join('data', 'train_labels.txt'), 'w') as f:              # train_labels text file로 저장
        for item in train_labels:
            f.write("%s\n" % item)

    with open(os.path.join('data', 'test_texts.txt'), 'w') as f:                # test_datasets text file로 저장
        for item in test_texts:
            f.write("%s\n" % item)

    with open(os.path.join('data', 'test_labels.txt'), 'w') as f:               # test_labels text file로 저장
        for item in test_labels:
            f.write("%s\n" % item)



if __name__ == "__main__":
    prepare_data()


