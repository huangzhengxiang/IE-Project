import torch
from model import simpleNet
from tqdm import tqdm

def evaluate(embed, model_name, test_iterator, ckpt_dir ,valid_iterator=None):
    """_summary_

    Args:
        embed (Embed): the ready-to-go word embedding
        test_iterator (Iterator): test_iterator
        valid_iterator (Iterator, optional): valid_iterator. Defaults to None.

    Returns:
        test_acc(float): accuracy for test iterator
        val_acc(float): accuracy for validation iterator
    """
    print("Evaluation Starts!")
    ckpt = torch.load(ckpt_dir)
    embed_dim = ckpt['embed_dim']
    h_dim = ckpt['h_dim']
    out_dim = ckpt['out_dim']
    model = simpleNet(embed_dim,h_dim,out_dim)
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    val_acc = None
    if valid_iterator != None:
        TP_val = []
        val_size = []
        with torch.no_grad():
            for point in tqdm(valid_iterator):
                if model_name=="simple":
                    x = embed(point.text.transpose(1,0))
                elif model_name=="bert":
                    x = embed(point.text.transpose(1,0))[0]
                pred = torch.round(torch.sigmoid(model(x).reshape(-1)))
                TP_val.append((pred==point.label).sum().float().item())
                val_size.append(pred.shape[0])
        val_acc = sum(TP_val) / float(sum(val_size))
    TP_test = []
    test_size = []
    with torch.no_grad():
        for point in tqdm(test_iterator):
            if model_name=="simple":
                x = embed(point.text.transpose(1,0))
            elif model_name=="bert":
                x = embed(point.text.transpose(1,0))[0]
            pred = torch.round(torch.sigmoid(model(x).reshape(-1)))
            TP_test.append((pred==point.label).sum().float().item())
            test_size.append(pred.shape[0])
    test_acc = sum(TP_test) / float(sum(test_size))
    return test_acc, val_acc