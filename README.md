
### download pdfs
    python download.py --meta_path=resources/parspec_test_data.csv --dst_dir_path=data/test --threads=25
    python download.py --meta_path=resources/parspec_train_data.csv --dst_dir_path=data/train --threads=100


### extract text from pdf
    python prep_data.py --meta_path=resources/parspec_test_data.csv --src_dir=data/test --dst_file_path=data/test-dataset.csv
    python prep_data.py --meta_path=resources/parspec_train_data.csv --src_dir=data/train --dst_file_path=data/train-dataset.csv

### Modelling
    The extracted text is used to finetune a DistilBertForSequenceClassification `distilbert-base-uncased-finetuned-sst-2-english` using LoRA.

### Report on the provided test set:

    Accuracy:   87.5 %
    TP: 16  | FN: 4
    FP: 6 	| TN: 54

### Running inference:
    
