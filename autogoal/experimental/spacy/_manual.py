import spacy
from spacy.util import minibatch, compounding
from spacy import displacy
from spacy.training import Example
import random
import pickle
from sys import argv
from pathlib import Path
from autogoal.kb import AlgorithmBase

from autogoal.grammar import CategoricalValue, BooleanValue
from autogoal.kb import Sentence, Word, FeatureSet, Seq
from autogoal.kb import Supervised
from autogoal.utils import nice_repr

@nice_repr
class SpacyTagger(AlgorithmBase):
    """
    This class is a Tokenizer using Spacy 3
    """
    def __init__(
        self,
        language: CategoricalValue("en", "es"),
        extract_pos: BooleanValue(),
        extract_lemma: BooleanValue(),
        extract_pos_tag: BooleanValue(),
        extract_dep: BooleanValue(),
        extract_entity: BooleanValue(),
        extract_details: BooleanValue(),
        extract_sentiment: BooleanValue(),
    ):
        self.language = language
        self.extract_pos = extract_pos
        self.extract_lemma = extract_lemma
        self.extract_pos_tag = extract_pos_tag
        self.extract_dep = extract_dep
        self.extract_entity = extract_entity
        self.extract_details = extract_details
        self.extract_sentiment = extract_sentiment
        self._nlp = None 


    def run(self, input: Sentence) -> Seq[FeatureSet]:
        
        if self._nlp == None:
            self._nlp = spacy.load(self.language)

        tokenized = self._nlp(input)
        self_tokenized = self.self_part_of_speech_tagger(tokenized)

        """
        Uncomment next line to show an entities visualizer
        Open browser localhost:5000
        """
        #self.self_displacy(tokenized)   

        return self_tokenized

    def self_part_of_speech_tagger(self, tokenized):

        flags = []

        for token in tokenized:
            token_flags = {}
            token_flags["text"] = token.text
            if self.extract_lemma:
                token_flags["lemma"] = token.lemma_
            if self.extract_pos_tag:
                token_flags["pos"] = token.pos_

                for kv in token.tag_.split("|"):
                    kv = kv.split("=")
                    if len(kv) == 2:
                        token_flags["tag_" + kv[0]] = kv[1]
                    else:
                        token_flags["tag_" + kv[0]] = True

            if self.extract_dep:
                token_flags["dep"] = token.dep_
            if self.extract_entity:
                token_flags["ent_type"] = token.ent_type_
                token_flags["ent_kb_id"] = token.ent_kb_id_
            if self.extract_details:
                token_flags["is_alpha"] = token.is_alpha
                token_flags["is_ascii"] = token.is_ascii
                token_flags["is_digit"] = token.is_digit
                token_flags["is_lower"] = token.is_lower
                token_flags["is_upper"] = token.is_upper
                token_flags["is_title"] = token.is_title
                token_flags["is_punct"] = token.is_punct
                token_flags["is_left_punct"] = token.is_left_punct
                token_flags["is_right_punct"] = token.is_right_punct
                token_flags["is_space"] = token.is_space
                token_flags["is_bracket"] = token.is_bracket
                token_flags["is_quote"] = token.is_quote
                token_flags["is_currency"] = token.is_currency
                token_flags["like_url"] = token.like_url
                token_flags["like_num"] = token.like_num
                token_flags["like_email"] = token.like_email
                token_flags["is_oov"] = token.is_oov
                token_flags["is_stop"] = token.is_stop
            if self.extract_sentiment:
                token_flags["sentiment"] = token.sentiment

            flags.append(token_flags)

        return flags

    def self_displacy(self, tokenized, style='dep'):

        displacy.serve(tokenized, style)
            
    
@nice_repr
class SpacyNER(AlgorithmBase):
    """
    This class is a Named Entity Recognition 

    """
    def __init__(self,
        language: CategoricalValue("en", "es"),
        save_model: BooleanValue(),
        output_dir = 'C:/Save_Model.model',
        ):
        self.language = language
        self.output_dir = output_dir
        self._nlp = spacy.load("en_core_web_sm") if self.language=="en" else spacy.load("es_core_news_sm") 

    """
    Comments:
    Run command to serch for entities in the input Sentence
    Parameters:
    input: Input text to recognize entities
    training_data_doc: Path to load a spacy's pretrained data
    input_new_labels: New entities labels to add to the model
    				  None default parameter if not needed to add entities


    Uncomment line 149 to show an entities visualizer
    Open browser localhost:5000
    """
    
    def run(self, input: Sentence, training_data_doc, input_new_labels=None) -> Seq[FeatureSet]:

        if input_new_labels != None:
            self.add_ent(training_data_doc, input_new_labels, self.output_dir)
        
        tokenized = self._nlp(input)
        
        #self.self_displacy(tokenized)

        return tokenized.ents

		
    """
    Comments:
    Add custom entity labels to train spacy models
    Parameters:
    training_data_doc: Path to load a spacy's pretrained data, in test.py: data/wikigold
    new_labels: New entities labels to add to the model
    output_dir: OutPut dir to save the model...If save_model ;)
    """
    def add_ent(self, training_data_doc, new_labels, output_dir, itns=1000):

        with open(training_data_doc, 'rb') as fp:
        	#test training_data = pickle.load(fp)
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

        if 'ner' not in self._nlp.pipe_names:
            ner = self._nlp.create_pipe('ner')
            self._nlp.add_pipe(ner)
        else: 
            ner = self._nlp.get_pipe('ner')

        for label in new_labels:
            ner.add_label(label)

        if self._nlp is None:
            optimizer = self._nlp.begin_training()
        else:
            optimizer = self._nlp.create_optimizer()

        #Get pipe names to disable during training
        pipes = [pipe for pipe in self._nlp.pipe_names if pipe != 'ner']
        with self._nlp.disable_pipes(*pipes):
            for itn in range(itns):
                random.shuffle(training_data)
                losses = {}
                batches = minibatch(training_data, size=compounding(1.,1., 0.))

                examples = []
                for batch in batches:
                    texts, annotations = zip(*batch)
                    if annotations[0] != 'O':
                        print(texts[0], annotations[0])
                        doc = self._nlp.make_doc(texts[0])
                        token_ref = [texts[0]]
                        tags_ref = [annotations[0]]                
                        examples.append(Example.from_dict(doc,{'words':token_ref, "tags":tags_ref}))
                self._nlp.update(examples, sgd=optimizer, drop=0.35, losses=losses)
        
        if self.save_model:
            self.save_model()

    def save_model(self, name='new_model'):

        dir = Path(self.output_dir)
        if not dir.exists():
            dir.mkdir()

        self._nlp.meta['name'] = "new_model_name"
        self._nlp.to_disk(dir)    

	# Displacy visualizer
    def self_displacy(self, tokenized, style='ent'):

        displacy.serve(tokenized, style)
