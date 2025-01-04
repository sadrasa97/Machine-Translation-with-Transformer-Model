# Machine Translation with Transformer Model

This project implements a machine translation system using a Transformer model to translate English to German based on the Multi30k dataset. The model is trained on the dataset, and evaluation is done using BLEU score for translation quality. This notebook covers the following steps:

1. **Dataset Loading**: Using `torchtext` to load the Multi30k dataset.
2. **Tokenization**: Implementing tokenizers using `spaCy` for both English and German.
3. **Vocabulary Building**: Building vocabulary for both the source (English) and target (German) languages.
4. **Dataset Preparation**: Creating a custom `Dataset` class for the translation task.
5. **Model Definition**: Defining the Transformer model using custom hyperparameters.
6. **Training**: Training the model on the translation task and plotting the training loss.
7. **Translation and Evaluation**: Using beam search for translation and evaluating the model using BLEU score.

## Requirements

To run this project, you'll need the following libraries:

- `torch`
- `torchtext`
- `spacy`
- `matplotlib`
- `nltk`
- `torch.utils.data`
- `torch.optim`
  
Make sure to install the required packages before running the notebook:

```bash
pip install torch torchtext spacy matplotlib nltk
```

You will also need to download the spaCy language models for English and German:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Steps

### 1. Dataset Loading

The Multi30k dataset is loaded using `torchtext` with custom URLs for the training, validation, and test splits.

```python
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"
multi30k.URL["test"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/mmt_task1_test2016.tar.gz"
```

### 2. Tokenization

The English and German text is tokenized using the respective spaCy models:

```python
spacy_eng = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]
```

### 3. Vocabulary Building

The vocabularies for both the source (English) and target (German) languages are built from the training dataset:

```python
SRC_VOCAB = build_vocab_from_iterator(yield_tokens(train_iter, SRC_TOKENIZER), specials=['<PAD>', '<UNK>', '<BOS>', '<EOS>'])
TGT_VOCAB = build_vocab_from_iterator(yield_tokens(train_iter, TGT_TOKENIZER), specials=['<PAD>', '<UNK>', '<BOS>', '<EOS>'])
```

### 4. Dataset Preparation

The custom `TranslationDataset` class prepares the dataset by tokenizing the sentences, adding special tokens (`<BOS>`, `<EOS>`), padding/truncating sentences, and converting tokens to indices.

### 5. Model Definition

The Transformer model is defined using custom hyperparameters. You can adjust the number of layers, hidden dimensions, and other configurations based on your needs.

```python
model = TransformerModel(src_vocab_size=src_vocab_size,
                         tgt_vocab_size=tgt_vocab_size,
                         N_e=8,
                         N_d=8,
                         d_model=1024,
                         h=8,
                         d_ff=2048,
                         max_len=50,
                         src_vocab=SRC_VOCAB,  
                         tgt_vocab=TGT_VOCAB) 
```

### 6. Training

The model is trained using the `CrossEntropyLoss` and `Adam` optimizer. During training, the loss is tracked, and the training loss curve is plotted:

```python
plt.plot(range(1, num_epochs + 1), loss_values, marker='o', color='b')
plt.title('Training Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
```

### 7. Translation and Evaluation

The model can be evaluated using beam search for translation. The BLEU score is calculated to evaluate the model's performance on the validation set.

```python
def translate_sentence(sentence, src_vocab, tgt_vocab, model, src_tokenizer, tgt_tokenizer, max_len=50, device='cpu'):
    ...
    
def evaluate_model(model, data_loader):
    ...
```

The `evaluate_model` function computes the BLEU score, and the translations are displayed for manual inspection.

## Evaluation Output

The model's translation performance is evaluated using BLEU score, and the output is printed for each sentence:

```python
Smooth BLEU Score: 0.4123
```

The translations and their corresponding references are also displayed.
