from typing import List
import xml.etree.cElementTree as ET
from xml.dom import minidom
import shutil
import os
import torch
from tqdm import tqdm
from torchtext.legacy import data
from torchtext.legacy import datasets
from transformers import BertTokenizer, BertModel
from model import simpleNet, Embed
import argparse
import pandas as pd

# review
# start_token = "<reviews>"
# med_token = "review"
# end_token = "</reviews>"

# weibo
start_token = "<weibos>"
med_token = "weibo"
end_token = "</weibos>"

def output_XML(id: List, text: List, labels: List, outdir):
    root_element = ET.Element("weibos")
    for i in range(len(id)):
        if labels[i]==1:
            subb = ET.SubElement(root_element, "weibo", attrib={"id": str(id[i]), "polarity": str(labels[i])})
        else:
            subb = ET.SubElement(root_element, "weibo", attrib={"id": str(id[i]), "polarity": str(-1)})
        subb.text = text[i]
    xml_string = ET.tostring(root_element)
    dom = minidom.parseString(xml_string)
    with open("output_old.xml", 'w+', encoding='utf-8') as f:
        dom.writexml(f, indent='', newl='\n',
                     addindent='\t', encoding='utf-8')
        f.seek(0)
        f.readline()
        target_file = open(outdir, 'wt', encoding="utf-8")
        shutil.copyfileobj(f, target_file)
        target_file.close()
    os.remove("output_old.xml")

def process(sent,lang):
    # if lang=="cn":
    #     if len(sent)<=20:
    #         sent = sent+sent
    if lang=="cn":
        tokenizer = BertTokenizer.from_pretrained("./ckpt/roberta/chinese-roberta-wwm-ext-large")
    elif lang=="en":
        tokenizer = BertTokenizer.from_pretrained("./ckpt/bert/bert-large-uncased")
    input_ids = torch.tensor(tokenizer(sent,
                        padding="max_length",
                        max_length=512,
                        truncation=True)['input_ids'])
    return input_ids

def predict(embed, model, text, tuning=False):
    with torch.no_grad():
        x = embed(text.reshape(1,-1)).last_hidden_state
        pred = torch.round(torch.sigmoid(model(x).reshape(-1)))
        return int(pred.item())

def parsetest(dir):
    olddir="tem_medium_file.xml"
    with open(olddir, "wt", encoding="utf-8") as old_f:
        with open(dir, "rt", encoding="utf-8") as f:
            content = f.read()
            content = content.replace("&", "and")
            if not content.strip().startswith(start_token):
                content = start_token+"\n"+ content+"\n"+end_token
            old_f.write(content)
    data = pd.read_xml(olddir)[med_token].tolist()
    ids =  pd.read_xml(olddir)["id"].tolist()
    info = []
    for i in tqdm(range(len(data))):
        info.append(data[i].replace("\n"," "))
    os.remove(olddir)
    return ids, info

if __name__=="__main__":
    # 
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",type=str)
    parser.add_argument("--ckpt",type=str)
    parser.add_argument("--lang",type=str)
    parser.add_argument("--input",type=str)
    parser.add_argument("--tuning",type=int,default=0)
    parser.add_argument("--output",type=str,default="task2output.xml")
    args=parser.parse_args()
    ckpt_dir=args.ckpt
    language=args.lang
    infile=args.input
    model_name=args.model
    tuning=bool(args.tuning)
    outdir=args.output
    # 
    ckpt = torch.load(ckpt_dir)
    embed_dim = ckpt['embed_dim']
    h_dim = ckpt['h_dim']
    out_dim = ckpt['out_dim']
    if tuning and model_name=="fc":
        model = torch.nn.Linear(embed_dim, out_dim)
    else:
        model = simpleNet(embed_dim, h_dim, out_dim)
    model.load_state_dict(ckpt['model'])
    if language=="en":
        embed = BertModel.from_pretrained("./ckpt/bert_tuned/bert-large-uncased")
    else:
        embed = BertModel.from_pretrained("./ckpt/roberta_tuned/chinese-roberta-wwm-ext-large")	
    embed_dim = embed.config.to_dict()['hidden_size']
    model.cuda()
    model.eval()
    embed.cuda()
    embed.eval()
    # 
    sent_ids, sent = parsetest(infile) # List, read from the input xml!
    results = []
    for idx in tqdm(range(len(sent))):
        sent_text = process(sent=sent[idx],lang=language).cuda()
        result = predict(embed=embed, model=model, text=sent_text, tuning=(tuning and model_name=="fc"))
        results.append(result)
    print("{0:.2f}% positive, {1:.2f}% negative.".format(100*sum(results)/len(results),100-100*sum(results)/len(results)))
    # 
    output_XML(sent_ids, sent, results, outdir)