AutoGOAL can be used directly from the CLI for some tasks. To see all available commands just run:

    python3 -m autogoal

## Inspect the contrib modules

To see all the contrib modules installed and the available algorithms, run:

    python3 -m autogoal contrib list

A fully installed version of AutoGOAL will show something like this:

    ⚙️  Found a total of 134 matching algorithms.
    🛠️  sklearn: 78 algorithms.
    🛠️  nltk: 26 algorithms.
    🛠️  gensim: 5 algorithms.
    🛠️  keras: 4 algorithms.
    🛠️  torch: 2 algorithms.
    🛠️  spacy: 1 algorithms.
    🛠️  wikipedia: 4 algorithms.
    🛠️  wrappers: 9 algorithms.
    🛠️  regex: 5 algorithms.

Use `--verbose` to actually list all the algorithms. Additionally, you can pass `--include` and `--exclude` to filter by algorithm name, and `--input` and/or `--output` to filter based on the input and output of each algorithm. These four parameters accept a regular expression. For example:

    $ python3 -m autogoal contrib list --exclude 'Tokenizer|Encoder' --input Sentence --verbose 
    
    ⚙️  Found a total of 7 matching algorithms.
    🛠️  sklearn: 4 algorithms.
     🔹 CountVectorizer           : List(Sentence()) -> MatrixContinuousSparse()
     🔹 CountVectorizerNoTokenize : List(Sentence()) -> MatrixContinuousSparse()
     🔹 HashingVectorizer         : List(Sentence()) -> MatrixContinuousSparse()
     🔹 TfidfVectorizer           : List(Sentence()) -> MatrixContinuousSparse()
    🛠️  torch: 1 algorithms.
     🔹 BertTokenizeEmbedding     : List(Sentence(language=english)) -> Tensor3()
    🛠️  spacy: 1 algorithms.
     🔹 SpacyNLP                  : Sentence() -> Tuple(List(Word()), List(Flags()))
    🛠️  wrappers: 1 algorithms.
     🔹 SentenceFeatureExtractor  : Sentence() -> Flags()
