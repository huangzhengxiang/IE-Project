from tqdm import tqdm
import pickle
import json
import argparse

data_dir="./data/CCFtrain02/en_sample_data/"
pos_file="sample.positive.txt"
neg_file="sample.negative.txt"
json_dir="./data/CCF/json/"
# need_pkl=True
need_pkl=False
need_json=True
# need_json=False

def parsingCCF(dir,file_type,split,need_pkl,need_json):
    info=[]
    with open(dir,"rt",encoding="utf-8") as f:
        content=f.read()
        content=content.split("\n")
        elem={}
        if split:
            for i in tqdm(range(len(content))):
                if content[i]=="":
                    continue
                if content[i][0]=="<":
                    if content[i][1]=="/":
                        elem={}
                        continue
                    else:
                        if need_pkl:
                            elem["id"]=int(content[i].split("\"")[1])
                            elem["label"]=file_type
                        if need_json:
                            elem["label"]=int(file_type)
                else:
                    elem["sent"]=content[i]
                    info.append(elem.copy())
        else:
            for i in tqdm(range(len(content))):
                if content[i]=="":
                    continue
                if content[i][0]=="<":
                    if content[i][1]=="/":
                        info.append(elem.copy())
                        elem={}
                    else:
                        if need_pkl:
                            elem["id"]=int(content[i].split("\"")[1])
                            elem["label"]=file_type
                        if need_json:
                            elem["label"]=int(file_type)
                else:
                    if not ("sent" in elem):
                        elem["sent"]=content[i]
                    else:
                        elem["sent"]=elem["sent"]+" "+content[i]
    return info

def dump_json(data,name):
    with open(json_dir+name,"wt") as f:
        for i in range(len(data)):
            json.dump(data[i],f)
            print("",file=f)

if __name__== "__main__":
    # 0. parsing
    parser=argparse.ArgumentParser()
    parser.add_argument("--pkl",type=int)
    parser.add_argument("--json",type=int)
    args=parser.parse_args()
    need_pkl=bool(args.pkl)
    need_json=bool(args.json)
    assert need_json != need_pkl
    data = []
    pos_dir=data_dir+pos_file
    neg_dir=data_dir+neg_file
    pos_info=parsingCCF(pos_dir,True,True,need_pkl,need_json)
    neg_info=parsingCCF(neg_dir,False,True,need_pkl,need_json)
    print(len(pos_info))
    print(len(neg_info))
    if need_pkl:
        with open("./data/CCF/pos_split.pkl","wb") as f:
            pickle.dump(pos_info,f)
        with open("./data/CCF/neg_split.pkl","wb") as f:
            pickle.dump(neg_info,f)
    if need_json:
        data.extend(pos_info)
        data.extend(neg_info)
        dump_json(data,"data_split.json")
    
    
    pos_info=parsingCCF(pos_dir,True,False,need_pkl,need_json)
    neg_info=parsingCCF(neg_dir,False,False,need_pkl,need_json)
    print(len(pos_info))
    print(len(neg_info))
    if need_pkl:
        with open("./data/CCF/pos_ori.pkl","wb") as f:
            pickle.dump(pos_info,f)
        with open("./data/CCF/neg_ori.pkl","wb") as f:
            pickle.dump(neg_info,f)
    data=[]
    if need_json:
        data.extend(pos_info)
        data.extend(neg_info)
        dump_json(data,"data_ori.json")