import re
import requests
import pathlib

DATASETS_REPO = 'https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/Datasets.md'
DATASETS = {}

r = requests.get(DATASETS_REPO)
src = r.content.decode()
pat = re.compile(r'<a href="(\S+)">Download</a>')

for url in pat.findall(src):
    name = pathlib.Path(url).parts[-2]
    DATASETS[name] = url
