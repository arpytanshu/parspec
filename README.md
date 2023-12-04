
### download pdfs
    python download.py --meta_path=resources/parspec_test_data.csv --dst_dir_path=data/test --threads=25
    python download.py --meta_path=resources/parspec_train_data.csv --dst_dir_path=data/train --threads=100


### extract text from pdf
    python prep_data.py --meta_path=resources/parspec_test_data.csv --src_dir=data/test --dst_file_path=data/test-dataset.csv
    python prep_data.py --meta_path=resources/parspec_train_data.csv --src_dir=data/train --dst_file_path=data/train-dataset.csv

### Modelling
    The extracted text is used to finetune a DistilBertForSequenceClassification `distilbert-base-uncased-finetuned-sst-2-english` using LoRA.

    The finetune.py script is used for finetuning.

### Report on the provided test set:

    Accuracy:   87.5 %
    TP: 16  | FN: 4
    FP: 6 	| TN: 54

### Running inference:
        
        
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    chkpt_path = '/content/parspec/checkpoints/checkpoint-270'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    tokenizer = AutoTokenizer.from_pretrained(chkpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(chkpt_path).to(device)
    model.eval()


    url = 'https://www.cooperlighting.com/api/assets/v1/file/CLS/content/347f567de4414421a1dcad3f014a0c77/corelite-continua-sq4-brochure'
    infer(model, tokenizer, url=url, dbg=True)

### Runnable Colab NB
`https://colab.research.google.com/drive/10aYmYGB5SQVU1ejdnnixjlIPlwtZFggm?usp=sharing`