import sys
sys.path.append('../models')
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from models.bert_fine_tuning import IMDbDataset, load_data

def evaluate_model(model_dir='results/fine_tuning'):
    train_texts, train_labels, test_texts, test_labels = load_data()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    test_dataset = IMDbDataset(test_texts, test_labels, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    training_args = TrainingArguments(
        output_dir='results/fine_tuning',
        per_device_eval_batch_size=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset
    )

    eval_results = trainer.evaluate()
    print(eval_results)

if __name__ == "__main__":
    evaluate_model()
