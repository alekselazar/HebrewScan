import urllib.request
from tqdm import tqdm

for i in tqdm(range(1, 5407), 'Downloading pages for dataset'):
    url = f'https://daf-yomi.com/Data/UploadedFiles/DY_Page/{i}.pdf'

    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    data = None
    with urllib.request.urlopen(req) as res:
        data = res.read()
    with open(f'pdfs/{i}.pdf', 'bw+') as file:
        file.write(data)

