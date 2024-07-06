# Sentiment Analysis with Various Tuning Methods

This project demonstrates sentiment analysis using the IMDB dataset with various model tuning methods. The tuning methods included are:
- Fine-Tuning
- Prompt Tuning
- P-Tuning
- Parameter-Efficient Fine-Tuning (PEFT)

## Project Structure

```plaintext
SentimentAnalysis/
├── data/
│   ├── prepare_data.py
│   ├── train_texts.txt
│   ├── train_labels.txt
│   ├── test_texts.txt
│   ├── test_labels.txt
│   ├── README.md
├── models/
│   ├── bert_fine_tuning.py
│   ├── prompt_tuning.py
│   ├── p_tuning.py
│   ├── peft.py
│   ├── README.md
├── results/
│   ├── prompt_tuning/
│   │   └── (results files)
│   ├── p_tuning/
│   │   └── (results files)
│   ├── fine_tuning/
│   │   └── (results files)
│   ├── peft/
│   │   └── (results files)
│   ├── README.md
├── scripts/
│   ├── train_prompt_tuning.py
│   ├── train_p_tuning.py
│   ├── train_fine_tuning.py
│   ├── train_peft.py
│   ├── evaluate_prompt_tuning.py
│   ├── evaluate_p_tuning.py
│   ├── evaluate_fine_tuning.py
│   ├── evaluate_peft.py
│   ├── README.md
├── main/
│   ├── main.py
│   ├── README.md
├── logs/
│   ├── prompt_tuning/
│   │   └── (log files)
│   ├── p_tuning/
│   │   └── (log files)
│   ├── fine_tuning/
│   │   └── (log files)
│   ├── peft/
│   │   └── (log files)
│   ├── README.md
├── README.md

```


## Getting Started
### Prerequisites
Make sure you have the following packages installed:

+ transformers
+ datasets
+ torch

You can install them using pip:

```
pip install transformers datasets torch
```


### Preparing the Data
1. Navigate to the data/ directory.
2. Run the prepare_data.py script to download and preprocess the IMDB dataset:

```
cd data
python prepare_data.py
```



This will generate the following files in the data/ directory:

+ train_texts.txt
+ train_labels.txt
+ test_texts.txt
+ test_labels.txt


### Training and Evaluation
Navigate to the scripts/ directory to train and evaluate models using different tuning methods.

#### Fine-Tuning
To train a model using fine-tuning:

```
cd scripts
python train_fine_tuning.py
```


To evaluate the fine-tuned model:

```
python evaluate_fine_tuning.py
```

#### Prompt Tuning
To train a model using prompt tuning:

```
python train_prompt_tuning.py
```

To evaluate the prompt-tuned model:

```
python evaluate_prompt_tuning.py
```

#### P-Tuning
To train a model using P-Tuning:

```
python train_p_tuning.py
```

To evaluate the P-Tuned model:

```
python evaluate_p_tuning.py
```

#### Parameter-Efficient Fine-Tuning (PEFT)
To train a model using PEFT:

```
python train_peft.py
```

To evaluate the PEFT model:

```
python evaluate_peft.py
```

### Logs and Results
+ __Logs__: The log files for each method are stored in the logs/ directory.
+ __Results__: The results for each method, including trained models and evaluation metrics, are stored in the results/ directory.



### Main Script
The main.py script in the main/ directory serves as the entry point for running and managing different tuning experiments.

### Contributing
Feel free to fork this repository and make improvements. Pull requests are welcome.
