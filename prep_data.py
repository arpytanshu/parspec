
from pathlib import Path
import fire
import PyPDF2
import pandas as pd
import sys

def progress_bar(current, total, bar_length=50, text="Progress"):
    anitext = ['\\', '|', '/', '-']
    percent = float(current) / total
    abs = f"{{{current} / {total}}}"
    arrow = '|' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    text = '[' + anitext[(current % 4)] + '] ' + text
    sys.stdout.write("\r{0}: [{1}] {2}% {3}".format(text, arrow + spaces, int(round(percent * 100)), abs))
    sys.stdout.flush()

def extract_text_from_pdf(pdf_path):
    pdf_file = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)    
    total_pages = len(pdf_reader.pages)
    text = ""

    for page_number in range(total_pages):
        page = pdf_reader.pages[page_number]
        text += page.extract_text()
    
    pdf_file.close()
    return text


def main(meta_path, src_dir, dst_file_path):
    meta_df = pd.read_csv(meta_path)
    extracted_text = []

    for ix, id in enumerate(meta_df.ID.tolist()):    
        if pd.isna(id):
            text = ' '
        else:
            try:
                src_file = Path(src_dir) / (id + '.pdf')
                text = extract_text_from_pdf(src_file)
            except:
                text = ' '
        
        extracted_text.append(text)
        progress_bar(ix+1, len(meta_df))

    meta_df['extracted_text'] = extracted_text
    meta_df.to_csv(dst_file_path, index=False, escapechar='\\')


if __name__ == '__main__':
    fire.Fire(main)

