# %%
import json
import pandas as pd

data = []

with open("../uci_datasets.log") as fp:
    for line in fp:
        item = json.loads(line)
        data.append(dict(
            dataset=item['dataset'],
            epoch=item['epoch'],
            score=(1-item['score'])*100.0
        ))

df = pd.DataFrame(data)

# %%
avg = df.groupby('dataset').agg(['mean', 'std'])
avg
