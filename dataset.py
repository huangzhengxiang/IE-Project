import torch
from tqdm import tqdm
from torchtext.legacy import data
from torchtext.legacy import datasets
from transformers import BertTokenizer
from model import simpleNet, Embed
import pickle
import random
class Tokenizer():
    def __init__(self,tokenizer,max_input_length) -> None:
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        
    def tokenize_and_cut(self, sentence):
        tokens = self.tokenizer.tokenize(sentence) 
        tokens = tokens[:self.max_input_length-2]
        return tokens

def build_data(dataset="ccf",
               lang="en",
               json_dir="./data/CCF/json/",
               vocab_size=int(1e5),
               vector_size=200,
               split_ratio=0.8,
               SEED=None):  
    if lang!="en":
        print("Language Not Supported Error!")
        exit(-1)        
    # 1. deterministic seed
    TEXT = data.Field(tokenize = 'spacy',
                  tokenizer_language = 'en_core_web_sm'
                  )
    LABEL = data.LabelField(dtype = torch.float)
    # 2. read in the data
    test_data=None
    if dataset=="ccf":
        fields = {'label': ('label', LABEL), 'sent': ('text', TEXT)}
        train_data = data.TabularDataset.splits(
                                path = json_dir,
                                train = 'endata.json',
                                format = 'json',
                                fields = fields
        )
        train_data = train_data[0]
        train_data, valid_data = train_data.split(split_ratio=split_ratio, random_state = random.seed(SEED))
    elif dataset=="imdb":
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        print(f'Number of testing examples: {len(test_data)}')
        train_data, valid_data = train_data.split(split_ratio=split_ratio, random_state = random.seed(SEED))
    print('Number of training examples: {0}'.format(len(train_data)))
    print('Number of validating examples: {0}'.format(len(valid_data)))
    # 3. build vocabulary
    TEXT.build_vocab(train_data, 
                max_size = vocab_size,
                vectors = 'glove.6B.{0}d'.format(vector_size)
                )
    LABEL.build_vocab(train_data)
    return TEXT, LABEL, train_data, valid_data, test_data

def build_data_bert(dataset="ccf",
                lang="en",
                json_dir="./data/CCF/json/",
                split_ratio=0.8,
                SEED=None):
    if lang=="en":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
    elif lang=="cn":
        if dataset!="ccf":
            print("Dataset Not Supported Error!")
            exit(-1)
        tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
        max_input_length = 512
    else:
        print("Language Not Supported Error!")
        exit(-1)
    init_token_idx = tokenizer.cls_token_id
    eos_token_idx = tokenizer.sep_token_id
    pad_token_idx = tokenizer.pad_token_id
    unk_token_idx = tokenizer.unk_token_id
    forBert=Tokenizer(tokenizer,max_input_length)
    TEXT = data.Field(use_vocab = False,
                  tokenize = forBert.tokenize_and_cut,
                  preprocessing = tokenizer.convert_tokens_to_ids,
                  init_token = init_token_idx,
                  eos_token = eos_token_idx,
                  pad_token = pad_token_idx,
                  unk_token = unk_token_idx)
    LABEL = data.LabelField(dtype = torch.float)
    # 2. read in the data
    test_data=None
    if dataset=="ccf":
        fields = {'label': ('label', LABEL), 'sent': ('text', TEXT)}
        train_data = data.TabularDataset.splits(
                                path = json_dir,
                                train = lang+'data.json',
                                format = 'json',
                                fields = fields
        )
        train_data = train_data[0]
        train_data, valid_data = train_data.split(split_ratio=split_ratio, random_state = random.seed(SEED))
    elif dataset=="imdb":
        train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
        print(f'Number of testing examples: {len(test_data)}')
        train_data, valid_data = train_data.split(split_ratio=split_ratio, random_state = random.seed(SEED))
    print('Number of training examples: {0}'.format(len(train_data)))
    print('Number of validating examples: {0}'.format(len(valid_data)))
    # 3. build vocabulary
    LABEL.build_vocab(train_data)
    return TEXT, LABEL, train_data, valid_data, test_data    