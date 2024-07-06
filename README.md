# DeepAdjust
🫛 Fine-Tuning for Sentiment Analysis 🫛
DeepAdjust is a comprehensive project aimed at performing advanced sentiment analysis using state-of-the-art tuning techniques on pre-trained language models. This project explores various methods such as fine-tuning, prompt tuning, p-tuning, and parameter-efficient fine-tuning (PEFT) to enhance model performance on sentiment analysis tasks.

&nbsp;&nbsp;
## Project Structure
DeepAdjust 프로젝트의 구조입니다. :)

```bash
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





&nbsp;&nbsp;
## Installation

To use this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/DeepAdjust.git
    ```
2. Navigate to the project directory:
    ```bash
    cd DeepAdjust
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


&nbsp;&nbsp;
## Usage

To run the main script, use the following command:

```bash
python main/main.py
