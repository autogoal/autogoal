# %%
import json
import pandas as pd
from pathlib import Path

from scipy.stats import ttest_ind_from_stats as ttest

# %%
data = []

with open(Path(__file__).parent / "uci_datasets.log") as fp:
    for line in fp:
        item = json.loads(line)
        data.append(dict(dataset=item["dataset"], score=(1 - item["score"]) * 100.0))

df = pd.DataFrame(data)
avg = df.groupby("dataset").agg(["mean", "std", "count"])
avg

# %%
algorithms = []
import re

algorithm_re = r"(\w*)\("
lines = []

for fname in ["haha", "meddocan", "uci"]:
    with open(Path(__file__).parent.parent / "logs" / f"{fname}_docker.log") as fp:
        inside = False
        for line in fp:
            if line.startswith("Pipeline("):
                inside = True

            if inside:
                lines.append((line, fname))

            if line.startswith(")") or line.endswith("])\n"):
                inside = False

for line, dataset in lines:
    for item in re.findall(algorithm_re, line):
        if item and item != "Pipeline":
            algorithms.append(dict(dataset=dataset, algorithm=item))


# %%
import random

data = []
problems = ["HAHA", "MEDDOCAN", "UCI"]
techs = ["NN", "Emb", "SC", "UN", "NLP"]


def is_nn(alg):
    return "Keras" in alg


def is_emb(alg):
    return "Embed" in alg


def is_sc(alg):
    return any(
        pattern in alg
        for pattern in ["Classifier", "Near", "NB", "SV", "Reg", "Percep"]
    )


def is_dr(alg):
    return any(
        pattern in alg
        for pattern in [
            "Mean",
            "K",
            "Kernel",
            "Transformer",
            "Affinity",
            "Scaler",
            "Encoder",
            "PCA",
            "ICA",
            "Birch",
            "NMF",
            "Isomap",
            "LatentDirichletAllocation",
            "Agglomeration",
            "Analysis",
        ]
    )


def is_nlp(alg):
    return any(pattern in alg for pattern in ["Tf", "Tagger", "Tokenize", "Vectorizer"])


def is_other(alg):
    return any(
        pattern in alg
        for pattern in [
            "Algorithm",
            "List",
            "MatrixContinuous",
            "Tuple",
            "Postag",
            "Sentence",
            "Builder",
            "ContinuousVector",
            "Word",
            "Tensor",
            "Categorical",
            "Aggregator",
        ]
    )


def categorize(alg):
    if is_nn(alg):
        return "NN"
    if is_emb(alg):
        return "Emb"
    if is_sc(alg):
        return "SC"
    if is_nlp(alg):
        return "NLP"
    if is_dr(alg):
        return "UN"
    if is_other(alg):
        return "Other"
    return None


for item in algorithms:
    cat = categorize(item["algorithm"])
    if cat in techs:
        data.append(dict(tech=cat, problem=item["dataset"].upper()))
    elif cat is None:
        print(item)

data = pd.DataFrame(data)

# %%
import altair as at

at.Chart(data).mark_bar().encode(
    x=at.X("tech", title="Type"),
    y=at.Y("count()", title="Total"),
    color=at.Color("problem", title="Dataset"),
).properties(width=200, height=120)


# %%
