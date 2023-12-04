
import pandas as pd
import requests
import os
import fire
from pathlib import Path
import multiprocessing
import sys


def download_link(link, dst_file):

    if os.path.exists(dst_file):
        return True
    else:
        try:
            response = requests.get(link)
            response.raise_for_status()
            
            with open(dst_file, 'wb') as file:
                file.write(response.content)
            print(f'Successfully downloaded {link}')
            return True
                
        except requests.exceptions.HTTPError as errh:
            print(f'Error downloading {link}: HTTP Error {errh}')
            return False

        except requests.exceptions.ConnectionError as errc:
            print(f'Error downloading {link}: Connection Error {errc}')
            return False

        except requests.exceptions.Timeout as errt:
            print(f'Error downloading {link}: Timeout Error {errt}')
            return False

        except requests.exceptions.RequestException as err:
            print(f'Error downloading {link}: {err}')
            return False


def progress_bar(current, total, bar_length=50, text="Progress"):
    anitext = ['\\', '|', '/', '-']
    percent = float(current) / total
    abs = f"{{{current} / {total}}}"
    arrow = '|' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    text = '[' + anitext[(current % 4)] + '] ' + text
    sys.stdout.write("\r{0}: [{1}] {2}% {3}".format(text, arrow + spaces, int(round(percent * 100)), abs))
    sys.stdout.flush()


def main(meta_path, dst_dir_path, threads=1):
    df = pd.read_csv(meta_path)
    
    if threads == 1:
        success_count = 0
        failure_count = 0
        for ix, row in df[['ID', 'URL']].iterrows():
            dst_path = Path(dst_dir_path) / (row.ID+'.pdf')
            if  download_link(row.URL, dst_path):
                success_count += 1
            else:
                failure_count += 1
            progress_bar(ix, len(df), text=f"#S:{success_count} F:{failure_count}")
        print(f'{success_count} success out of {len(df)}')
        
    else:
        pool = multiprocessing.Pool(processes=threads)
        urls = df.URL.tolist()[::-1]
        dst_files = df.ID.apply(lambda x: str(Path(dst_dir_path) / (str(x)+'.pdf'))).tolist()[::-1]
        pool.starmap(download_link, [(url, dst_file) for url, dst_file in zip(urls, dst_files)])
        pool.close()
        pool.join()


if __name__ == '__main__':
    fire.Fire(main)

