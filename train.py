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
import random

tuning = False
tune_prob = 0.5

json_dir="./data/CCF/json/"
vocab_size = int(1e5)
vector_size = 200
split_ratio=0.9
split_sent=False
hidden_dim = 64
out_dim = 1
batch_size = 32
lr = 5e-4
weight_decay = 0.0
nepoch = 5

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model,embed_dim,h_dim,out_dim,ckpt_dir):
    state_dict=dict(
        model=model.state_dict(),
        embed_dim=embed_dim,
        h_dim=h_dim,
        out_dim=out_dim,
    )
    torch.save(state_dict,ckpt_dir)

if __name__== "__main__":
    # 0. parsing
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str)
    parser.add_argument("--model",type=str)
    parser.add_argument("--lang",type=str)
    args=parser.parse_args()
    dataset=args.dataset
    model_name=args.model
    language=args.lang
    print("Start training on {} dataset".format(dataset))
    ckpt_dir="./ckpt/{}_{}_{}_best.pth".format(dataset,model_name,language)
    # 1. deterministic seed
    SEED = 1924
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # 2. data and vocabulary
    if model_name=="simple":
        TEXT, LABEL, train_data, valid_data, test_data = build_data(dataset=dataset,
                    lang=language,
                    json_dir="./data/CCF/json/",
                    vocab_size=vocab_size,
                    vector_size=vector_size,
                    split_ratio=split_ratio,
                    SEED=SEED)
    elif model_name=="bert":
        TEXT, LABEL, train_data, valid_data, test_data = build_data_bert(dataset=dataset,
            lang=language,
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
        if language=="en":
            embed = BertModel.from_pretrained('bert-base-uncased')
        else:
            embed = BertModel.from_pretrained("./ckpt/roberta/chinese-roberta-wwm-ext-large")	
        embed_dim = embed.config.to_dict()['hidden_size']
    # 5. build network
    model = simpleNet(embed_dim, hidden_dim, out_dim)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    print('The model has {0} trainable parameters'.format(count_parameters(model)))
    embed = embed.to(device)
    model = model.to(device)
    criterion = criterion.to(device)
    # 6. start training
    best_loss = 10.0
    for i in range(nepoch):
        train_loss = []
        train_size = []
        val_loss = []
        val_size = []
        for point in tqdm(train_iterator):
        # for point in train_iterator:
            optimizer.zero_grad()
            if tuning and (random.random()>tune_prob):
                # fine tune with 50%
                if model_name=="simple":
                    x = embed(point.text.transpose(1,0))
                elif model_name=="bert":
                    x = embed(point.text.transpose(1,0))[0]
            else:
                with torch.no_grad():
                    if model_name=="simple":
                        x = embed(point.text.transpose(1,0))
                    elif model_name=="bert":
                        x = embed(point.text.transpose(1,0))[0]
            pred = model(x).reshape(-1)
            loss = criterion(pred,point.label)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_size.append(pred.shape[0])
        with torch.no_grad():
            for point in tqdm(valid_iterator):
                if model_name=="simple":
                    x = embed(point.text.transpose(1,0))
                elif model_name=="bert":
                    x = embed(point.text.transpose(1,0))[0]
                pred = model(x).reshape(-1)
                loss = criterion(pred,point.label)
                val_loss.append(loss.item())
                val_size.append(pred.shape[0])
        t_loss = torch.matmul(torch.tensor(train_loss),torch.tensor(train_size).float()) / float(torch.tensor(train_size).sum().item())
        v_loss = torch.matmul(torch.tensor(val_loss),torch.tensor(val_size).float()) / float(torch.tensor(val_size).sum().item())
        if (v_loss < best_loss):
            best_loss = v_loss
            save_model(model, embed_dim, hidden_dim, out_dim, ckpt_dir)
            embed.save_pretrained("./ckpt/roberta_tuned/chinese-roberta-wwm-ext-large")
        print("epoch: {0}, train loss: {1:.4f}, val loss: {2:.4f}".format(i+1,t_loss.item(),v_loss.item()))

    # 7. evaluation
    train_acc, _ = evaluate(embed, model_name, train_iterator, ckpt_dir)
    print("train accuracy: {0:.2f}%".format(train_acc*100))        
    val_acc, _ = evaluate(embed, model_name, valid_iterator, ckpt_dir)
    print("validation accuracy: {0:.2f}%".format(val_acc*100))
    if dataset != "ccf":
        test_acc, _ = evaluate(embed, model_name, test_iterator, ckpt_dir)
        print("test accuracy: {0:.2f}%".format(test_acc*100))