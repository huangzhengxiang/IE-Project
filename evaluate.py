import nltk
import os
import torch
from tqdm import tqdm
from torchtext.legacy import data
from torchtext.legacy import datasets
from model import simpleNet, Embed
import pickle
import argparse
from dataset import build_data, build_data_bert
from utils.eval import evaluate
from transformers import BertTokenizer, BertModel

json_dir="./data/CCF/json/"
vocab_size=int(1e5)
vector_size=200
split_ratio=0.8
split_sent=False
batch_size=64

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__== "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--model",type=str)
    parser.add_argument("--ckpt",type=str)
    args=parser.parse_args()
    dataset=args.dataset
    model_name=args.model
    ckpt_dir=args.ckpt
    print("Start testing on {} dataset".format(dataset))
    # 1. deterministic seed
    SEED = 3428
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # 2. data and vocabulary
    if model_name=="simple":
        TEXT, LABEL, train_data, valid_data, test_data = build_data(dataset=dataset,
                    json_dir="./data/CCF/json/",
                    vocab_size=vocab_size,
                    vector_size=vector_size,
                    split_ratio=split_ratio,
                    SEED=SEED)
    elif model_name=="bert":
        TEXT, LABEL, train_data, valid_data, test_data = build_data_bert(dataset=dataset,
            json_dir="./data/CCF/json/",
            split_ratio=split_ratio,
            SEED=SEED)
    # 3. prepare for train
    device = 'cuda'
    if dataset=="ccf":
        train_iterator, valid_iterator = data.BucketIterator.splits(
            datasets=(train_data, valid_data), 
            batch_sizes = (batch_size,1),
            sort = False,
            device = device)
    elif dataset=="imdb":
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            datasets=(train_data, valid_data, test_data), 
            batch_sizes = (batch_size,batch_size,1),
            sort = False,
            device = device)
    # 4. using pretrained embedding!
    if model_name=="simple":
        input_dim = len(TEXT.vocab)
        embed_dim = vector_size
        unk_idx = TEXT.vocab.stoi[TEXT.unk_token]
        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
        embed = Embed(input_dim, embed_dim, pad_idx=pad_idx)
        embed.embed.weight.data.copy_(TEXT.vocab.vectors)
        embed.embed.weight.data[unk_idx] = torch.zeros(embed_dim).to(device)
        embed.embed.weight.data[pad_idx] = torch.zeros(embed_dim).to(device)
    elif model_name=="bert":
        embed = BertModel.from_pretrained('bert-base-uncased')
        embed_dim = embed.config.to_dict()['hidden_size']
    embed.to(device)
    # 5. Start testing
    train_acc, _ = evaluate(embed, model_name, train_iterator, ckpt_dir)
    print("train accuracy: {0:.2f}%".format(train_acc*100))        
    val_acc, _ = evaluate(embed, model_name, valid_iterator, ckpt_dir)
    print("validation accuracy: {0:.2f}%".format(val_acc*100))
    if dataset != "ccf":
        test_acc, _ = evaluate(embed, model_name, test_iterator, ckpt_dir)
        print("test accuracy: {0:.2f}%".format(test_acc*100))