import re
import requests
import pathlib
import pathlib

DATASETS_REPO = (
    "https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/Datasets.md"
)
SAVE_AT = pathlib.Path("~").expanduser() / ".autogoal" / "data"
DATASETS = {}

if not SAVE_AT.exists():
    SAVE_AT.mkdir()

file = pathlib.Path(SAVE_AT) / "datasets.md"

if not file.exists():
    r = requests.get(DATASETS_REPO)

    with file.open("w") as f:
        f.write(r.content.decode())

with file.open() as f:
    src = f.read()

pat = re.compile(r'<a href="(\S+)">Download</a>')

for url in pat.findall(src):
    name = pathlib.Path(url).parts[-2]
    DATASETS[name] = url
