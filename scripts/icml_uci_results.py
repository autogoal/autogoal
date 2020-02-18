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
best_values = dict(
    cars=(0.34, 0.51),
    german_credit=(23.91, 0.22),
    abalone=(73.14, 1.02),
    shuttle=(0.01, 0.01),
    yeast=(38.47, 2.36),
    dorothea=(6.02, 1.01),
    gisette=(2.24, 0.33),
)

worst_values = dict(
    cars=(1.38, 0.67),
    german_credit=(26.50, 2.32),
    abalone=(82.92, 8.38),
    shuttle=(0.12, 0.06),
    yeast=(40.51, 2.17),
    dorothea=(8.69, 1.54),
    gisette=(3.90, 0.40),
)

for row in avg.iterrows():
    dataset = row[0]
    mean = row[1][0]
    std = row[1][1]

    best_mean, best_std = best_values[dataset]
    worst_mean, worst_std = worst_values[dataset]

    _, p1 = ttest(mean, std, 20, best_mean, best_std, 20)
    _, p2 = ttest(mean, std, 20, worst_mean, worst_std, 20)

    if mean < best_mean and p1 < 0.05:
        mark = "<"
    elif mean > worst_mean and p2 < 0.05:
        mark = ">"
    else:
        mark = "~"

    print("%s: (%.3f, %.3f)%s" % (dataset, p1, p2, mark))

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
