

#%%
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import BitsAndBytesConfig
from transformers import TrainerCallback

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


class customDataset(Dataset):
    def __init__(self, df, tokenizer, max_tokens):
        super().__init__()

        fix_label = lambda x: {'yes':1, 'no':0, 0:0, 1:1}[x.lower() if type(x)==str else x]
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.df = df
        self.y = df['Is lighting product?'].apply(fix_label).values

    def __getitem__(self, index):
        row = self.df.iloc[index]
        string = row.extracted_text
        input_ids = self.tokenizer.encode(string)
        input_ids = input_ids[:min(len(input_ids), self.max_tokens)]
        labels = self.y[index]
        return {'input_ids':input_ids, 'labels':labels}
    
    def __len__(self):
        return len(self.y)
    
    def get_num_tokens(self, string):
        return len(self.tokenizer.encode(string))


class Collater:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        lengths = [len(x['input_ids']) for x in batch]
        max_length = max(lengths)
        
        input_ids = []
        attention_mask = []
        for ix, sample in enumerate(batch):
            input_ids.append([self.pad_id] * (max_length - lengths[ix]) + sample['input_ids'])
            attention_mask.append([0] * (max_length - lengths[ix]) + [1] * len(sample['input_ids']))
        
        labels = [x['labels'] for x in batch]
        
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        attention_mask = torch.tensor(attention_mask)
        
        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

MODEL_STRING = 'distilbert-base-uncased-finetuned-sst-2-english'
# MODEL_STRING = 'distilbert-base-uncased'

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # load_4bit_use_double_quant=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=6,
    lora_alpha=32,
    target_modules=["q_lin", "k_lin", "v_lin", "out_lin"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_STRING)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_STRING,
                                                            trust_remote_code=True,
                                                            quantization_config=bnb_config)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
max_tokens = model.config.max_position_embeddings



tr_df = pd.read_csv('/home/ansh/Desktop/Parspec/data/train-dataset-emb.csv')
te_df = pd.read_csv('/home/ansh/Desktop/Parspec/data/test-dataset-emb.csv')

tr_ds = customDataset(tr_df, tokenizer, max_tokens)
te_ds = customDataset(te_df, tokenizer, max_tokens)

collate_fn=Collater(pad_id=tokenizer.pad_token_id)


class MyCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"
    def __init__(self, eval_dataset, collate_fn):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn

    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        acc = self._evaluate(model)
        print('Evaluation:', acc)

    def _evaluate(self, model):
        print("running custom_evaluate...")
        model.eval()
        correct = 0;
        tp = 0;
        tn = 0;
        fp = 0;
        fn = 0;
        with torch.no_grad():
            for batch in DataLoader(self.eval_dataset, batch_size=128, collate_fn=self.collate_fn):
                outputs = model(**batch)
                preds = outputs.logits.argmax(1)
                labels = batch['labels']
                correct += (preds == labels).sum().item()
                tp += ((preds == 1) & (labels == 1)).sum()
                tn += ((preds == 0) & (labels == 0)).sum()
                fp += ((preds == 1) & (labels == 0)).sum()
                fn += ((preds == 0) & (labels == 1)).sum()
        acc = correct / len(self.eval_dataset)
        model.train()
        return {'accuracy': acc, 'tp': tp.item(), 'tn': tn.item(), 'fp': fp.item(), 'fn': fn.item()}
    

training_args = transformers.TrainingArguments(
    # auto_find_batch_size=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=128,
    num_train_epochs=3,
    learning_rate=1e-4,
    bf16=True,
    save_total_limit=5,
    logging_steps=10,
    eval_steps=10,
    output_dir='checkpoints/',
    save_strategy='steps',
    save_steps=10,
    resume_from_checkpoint=True,
)


trainer = transformers.Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tr_ds,
    eval_dataset=te_ds,
    args=training_args,
    data_collator=Collater(pad_id=tokenizer.pad_token_id),
    callbacks=[MyCallback(te_ds, collate_fn)]
)



trainer.train()
# trainer.evaluate()
# trainer.train(resume_from_checkpoint=True)


# %%
