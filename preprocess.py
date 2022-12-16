import pandas as pd
from tqdm import tqdm
import json
import argparse
import os
import shutil

data_dir = "./data/CCFtrain02/"
json_dir = "./data/CCF/json/"

need_json=True

def parsingCCF(dir, file_type):
    txtdir = dir
    xmldir = dir
    xmldir = dir.replace("txt", "xml")
    with open(xmldir, "wt", encoding="utf-8") as xml_f:
        with open(txtdir, "rt", encoding="utf-8") as f:
            content = f.read()
            content = content.replace("&", "and")
            if not content.strip().startswith("<reviews>"):
                content = "<reviews>\n" + content + "\n</reviews>"
            xml_f.write(content)
    data = pd.read_xml(xmldir)["review"].tolist()
    info = []
    for i in tqdm(range(len(data))):
        elem = {}
        elem["label"] = int(file_type)
        elem["sent"] = data[i].replace("\n"," ")
        info.append(elem)
    return info


def dump_json(data,name):
    with open(json_dir+name,"wt", encoding='utf-8') as f:
        for i in range(len(data)):
            json.dump(data[i],f, ensure_ascii=False)
            print("",file=f)


if __name__ == "__main__":
    # 0. parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=int, default= 1)
    parser.add_argument("--cnpath", type=str, default="cn_sample_data/")
    parser.add_argument("--enpath", type=str, default="en_sample_data/")
    args = parser.parse_args()
    need_json = bool(args.json)
    cnpath = args.cnpath
    enpath = args.enpath
    if os.path.exists(data_dir + enpath + "sample.positive.txt"):
        pos_file = "sample.positive.txt"
        neg_file = "sample.negative.txt"
    cndata = []
    endata = []
    cn_pos_dir = data_dir + cnpath + pos_file
    cn_neg_dir = data_dir + cnpath + neg_file
    en_pos_dir = data_dir + enpath + pos_file
    en_neg_dir = data_dir + enpath + neg_file
    print("parsing cn")
    cn_pos_info = parsingCCF(cn_pos_dir, True)
    cn_neg_info = parsingCCF(cn_neg_dir, False)
    cndata.extend(cn_pos_info)
    cndata.extend(cn_neg_info)
    dump_json(cndata, "cndata.json")
    print("parsing en")
    en_pos_info = parsingCCF(en_pos_dir, True)
    en_neg_info = parsingCCF(en_neg_dir, False)
    endata.extend(en_pos_info)
    endata.extend(en_neg_info)
    dump_json(endata, "endata.json")

