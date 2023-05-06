import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchtext.legacy import data
from torchtext.legacy import datasets
from model import simpleNet, Embed
import pickle
import argparse
from dataset import build_data, build_data_bert
from utils.eval import evaluate
from transformers import BertTokenizer, BertModel
from utils.mixout_tuning import adding_mixout
import random

json_dir="./data/CCF/json/"
vocab_size = int(1e5)
vector_size = 200
# dataset split ratio
# split_ratio=0.8
# # model
# hidden_dim = 64
out_dim = 1
# bs lr wd epoch
# batch_size = 10
# lr = 1e-3
weight_decay = 0.0
nepoch = 3


def create_params(seed=1):
    rnd = np.random.RandomState(seed)
    split_ratio = [0.8, 0.85 ,0.9][rnd.randint(0,3)]
    hidden_dim = [32, 64][rnd.randint(0, 2)]
    batch_size = rnd.randint(5, 11)
    lr = rnd.randint(1,10)*np.logspace(-5,-2,4)[rnd.randint(0,4)]
    mix_out = rnd.randint(0,2)
    mixout_prob = [0.75, 0.8, 0.85, 0.9][rnd.randint(0,4)]
    optim = ["adam", "adamw"][rnd.randint(0,2)]
    return split_ratio, hidden_dim, batch_size, lr, mix_out, mixout_prob, optim



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
    parser.add_argument("--tuning",type=int,default=0)
    parser.add_argument("--mixout",type=int,default=0)
    parser.add_argument("--pid",type=int,default=0)
    args=parser.parse_args()
    dataset=args.dataset
    model_name=args.model
    language=args.lang
    tuning=args.tuning
    mix_out=args.mixout
    pid=args.pid
    print("Start training on {} dataset".format(dataset))
    ckpt_dir="./ckpt/{}_{}_{}_best{}.pth".format(dataset,model_name,language,pid) # change
    # 1. deterministic seed
    SEED = 1924
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    max_search = 30
    search_history = {"split_ratio":[],"hidden_dim":[],"batch_size":[],"lr":[],"mixout":[],
                    "mixout_prob":[],"optim":[],"train_acc":[],"eval_acc":[]}

    for i in range(max_search):
        split_ratio, hidden_dim, batch_size, lr, mix_out, mixout_prob, optim = create_params(10*i+pid+1)
        if mix_out:
            batch_size = int(batch_size/2)
        print(split_ratio, hidden_dim, batch_size, lr, mix_out, mixout_prob, optim)
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
            if not tuning:
                batch_size=64
            train_iterator, valid_iterator = data.BucketIterator.splits(
                datasets=(train_data, valid_data), 
                batch_sizes = (batch_size,1),
                sort = False,
                device = device)
        elif dataset=="imdb":
            batch_size=64
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
        elif tuning or model_name=="bert":
            if language=="en":
                embed = BertModel.from_pretrained('./ckpt/bert/bert-large-uncased')
            else:
                embed = BertModel.from_pretrained("./ckpt/roberta/chinese-roberta-wwm-ext-large")
                if mix_out:
                    embed = adding_mixout(embed, mixout_prob)
            embed_dim = embed.config.to_dict()['hidden_size']
        # 5. build network
        if tuning and model_name=="fc":
            model = torch.nn.Linear(embed_dim, out_dim)
        else:
            model = simpleNet(embed_dim, hidden_dim, out_dim)
        if optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[1,2,3],gamma=0.4)
        criterion = torch.nn.BCEWithLogitsLoss()
        print('The model has {0} trainable parameters'.format(count_parameters(model)))

        # single GPU
        embed = embed.to(device)
        model = model.to(device)
        criterion = criterion.to(device)
        
        # multi GPU
        # if not tuning:
        #     embed = embed.to(device)
        #     model = model.to(device)
        #     criterion = criterion.to(device)
        # else:
        #     device_ids = [0, 1, 6]
        #     embed = torch.nn.DataParallel(embed, device_ids=device_ids)
        #     model = torch.nn.DataParallel(model, device_ids=device_ids)
        #     embed = embed.cuda(device=device_ids[0])
        #     model = model.cuda(device=device_ids[0])

        # 6. start training
        best_loss = 10.0
        if model_name=="simple":
            nepoch=10
        for i in range(nepoch):
            train_loss = []
            train_size = []
            val_loss = []
            val_size = []
            for point in tqdm(train_iterator):
            # for point in train_iterator:
                optimizer.zero_grad()
                if tuning:
                    if model_name=="fc":
                        x = embed(point.text.transpose(1,0)).last_hidden_state[:,0]
                    else:
                        x = embed(input_ids=point.text.transpose(1,0)).last_hidden_state
                else:
                    with torch.no_grad():
                        if model_name=="simple":
                            x = embed(point.text.transpose(1,0))
                        elif model_name=="bert":
                            x = embed(point.text.transpose(1,0)).last_hidden_state
                pred = model(x).reshape(-1)
                loss = criterion(pred,point.label)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                train_size.append(pred.shape[0])
            scheduler.step()
            with torch.no_grad():
                for point in tqdm(valid_iterator):
                    if model_name=="simple":
                        x = embed(point.text.transpose(1,0))
                    elif tuning and model_name=="fc":
                        x = embed(point.text.transpose(1,0)).last_hidden_state[:,0]
                    elif model_name=="bert":
                        x = embed(point.text.transpose(1,0)).last_hidden_state
                    pred = model(x).reshape(-1)
                    loss = criterion(pred,point.label)
                    val_loss.append(loss.item())
                    val_size.append(pred.shape[0])
            t_loss = torch.matmul(torch.tensor(train_loss),torch.tensor(train_size).float()) / float(torch.tensor(train_size).sum().item())
            v_loss = torch.matmul(torch.tensor(val_loss),torch.tensor(val_size).float()) / float(torch.tensor(val_size).sum().item())
            if (v_loss < best_loss):
                best_loss = v_loss

                # single GPU
                save_model(model, embed_dim, hidden_dim, out_dim, ckpt_dir)
                if tuning and language=="cn":
                    embed.save_pretrained("./ckpt/roberta_tuned/chinese-roberta-wwm-ext-large{}".format(pid)) # change
                elif tuning and language=="en":
                    embed.save_pretrained("./ckpt/bert_tuned/bert-large-uncased")

                # multi GPU
                # save_model(model.module, embed_dim, hidden_dim, out_dim, ckpt_dir)
                # if tuning and language=="cn":
                #     embed.module.save_pretrained("./ckpt/roberta_tuned/chinese-roberta-wwm-ext-large")
                # elif tuning and language=="en":
                #     embed.module.save_pretrained("./ckpt/bert_tuned/bert-large-uncased")
            print("epoch: {0}, train loss: {1:.4f}, val loss: {2:.4f}".format(i+1,t_loss.item(),v_loss.item()))

        # 7. evaluation
        train_acc, _ = evaluate(embed, model_name, train_iterator, ckpt_dir, tuning=tuning)
        print("train accuracy: {0:.2f}%".format(train_acc*100))     
        val_acc, _ = evaluate(embed, model_name, valid_iterator, ckpt_dir, tuning=tuning)
        print("val accuracy: {0:.2f}%".format(val_acc*100))
        print("\n")
        if dataset != "ccf":
            test_acc, _ = evaluate(embed, model_name, test_iterator, ckpt_dir, tuning=tuning)
            print("test accuracy: {0:.2f}%".format(test_acc*100))
        
        search_history["split_ratio"].append(split_ratio)
        search_history["hidden_dim"].append(hidden_dim)
        search_history["batch_size"].append(batch_size)
        search_history["lr"].append(lr)
        search_history["mixout"].append(mix_out)
        search_history["mixout_prob"].append(mixout_prob)
        search_history["optim"].append(optim)
        search_history["train_acc"].append("{0:.2f}%".format(train_acc*100))
        search_history["eval_acc"].append("{0:.2f}%".format(val_acc*100))

    df=pd.DataFrame(search_history)
    df.sort_values(by="eval_acc" , inplace=True, ascending=False)
    df.to_csv("./train_output/search_history{}.csv".format(pid))

        