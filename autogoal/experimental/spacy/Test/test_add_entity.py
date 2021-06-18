import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import random
import pickle
from pathlib import Path
from sys import argv

Labels = [ 'I-per', 'B-per', 'I-org', 'B-org', 'I-gpe' ] #set new entities tags

with open(argv[1], 'rb') as fp:
    training_data = []
    lines = fp.readlines()
    for l in lines:
        if l != b'\n':
            try:
                s = l.decode('utf-8')
                s = s.split()          
                training_data.append((s[0],s[1]))
            except:
                continue

nlp = spacy.load("en_core_web_sm")

if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
else: 
    ner = nlp.get_pipe('ner')

for label in Labels:
    ner.add_label(label)

optimizer = nlp.create_optimizer()

pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*pipes):
    for itn in range(1000):
        random.shuffle(training_data)
        losses = {}
        batches = minibatch(training_data, size=compounding(1.,1., 0.))

        examples = []
        for batch in batches:
            texts, annotations = zip(*batch)
            if annotations[0] != 'O':
                print(texts[0], annotations[0])
                doc = nlp.make_doc(texts[0])
                token_ref = [texts[0]]
                tags_ref = [annotations[0]]                
                examples.append(Example.from_dict(doc,{'words':token_ref, "tags":tags_ref}))
        nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)


with open(argv[2], 'r') as f:
    doc = f.readlines()
    for line in doc:
        print(line)
        doc = nlp(line)
        for ent in doc.ents:
            print(ent.label_, ent.text) #test not sure it works

