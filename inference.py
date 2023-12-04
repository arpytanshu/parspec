
#%%

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np

from pathlib import Path

from download import download_link
from prep_data import extract_text_from_pdf



def infer(model, tokenizer, url=None, dst_file=None, dbg=True):

    if (url == None) and (dst_file == None):
        print('No url or dst_file provided. Exiting')
        return

    if url != None:
        dst_file = 'temp.pdf'
        if os.path.exists(dst_file):
            os.remove(dst_file)
        
        if download_link(url, dst_file):
            if dbg: print('Download successful')

    try:
        text = extract_text_from_pdf(dst_file)
        if dbg: print('Text extraction successful')
    except:
        text = ' '
        if dbg: print('Text extraction from pdf failed')

    input_ids = tokenizer.encode(text)
    input_ids = input_ids[:min(len(input_ids), model.config.max_position_embeddings)]
    input_ids = torch.tensor(input_ids).view(1, -1)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():

        outputs = model(input_ids)
    class_ix = outputs.logits.argmax(1).item()
    nl_class = {1: 'Is lighting product? YES', 0: 'Is lighting product? NO'}[class_ix]
    if dbg: print(nl_class)

    if os.path.exists('temp.pdf'):
        os.remove('temp.pdf')

    return class_ix


def evaluate(chkpt_path, files_basepath, test_meta_df='data/test-dataset.csv'):
    
    tokenizer = AutoTokenizer.from_pretrained(chkpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(chkpt_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    files_basepath = Path(files_basepath)
    test_meta_df = pd.read_csv(test_meta_df)

    preds = []
    labels = []

    for ix, row in test_meta_df.iterrows():
        file_name = row.ID + '.pdf'
        file_path = files_basepath / file_name
        class_ix = infer(model, tokenizer, dst_file=str(file_path), dbg=False)
        # class_ix = infer(model, tokenizer, url=row.URL, dbg=False)
        true_label = row['Is lighting product?']
        # print(f'{ix} - {class_ix} - {true_label}')
        preds.append(class_ix)
        labels.append(true_label)

    preds = np.array(preds)
    labels = np.array(labels)

    correct = (preds == labels).sum().item()
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    print(f'Accuracy: {correct / len(labels)}')

    print(f'TP: {tp} \t| FN: {fn}')
    print(f'FP: {fp} \t| TN: {tn}') 

#%%


evaluate(chkpt_path='checkpoints/checkpoint-270', files_basepath='data/test/')


# %%

url = 'https://www.cooperlighting.com/api/assets/v1/file/CLS/content/347f567de4414421a1dcad3f014a0c77/corelite-continua-sq4-brochure'
infer(model, tokenizer, url=url)


