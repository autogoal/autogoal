AutoGOAL can be used directly from the CLI for some tasks. To see all available commands just run:

    autogoal

![](autogoal_cli.svg)

## Run the streamlit demo

Just run:

    autogoal demo

## Inspect the contrib modules

To see all the contrib modules installed and the available algorithms, run:

    autogoal contrib list

A fully installed version of AutoGOAL will show something like this:

![](autogoal_cli_contrib.svg)

Use `--verbose` to actually list all the algorithms. Additionally, you can pass `--include` and `--exclude` to filter by algorithm name, and `--input` and/or `--output` to filter based on the input and output of each algorithm. These four parameters accept a regular expression. For example:

    $ autogoal contrib list --exclude 'Tokenizer|Encoder' --input Sentence --verbose 

Will show all algorithms whose names don't match with `Tokenizer` or `Encoder`, and whose inputs match with `Sentence`.

![](autogoal_cli_contrib_list.svg)

## Fit and predict with an AutoML model

To fit an AutoML model on a custom dataset, run:

    autogoal ml fit <INPUT>

The `INPUT` parameter should be a dataset. Two options are available, CSV and JSON files. You can also configure several search parameters, including the total search time, the available memory, the number of iterations, etc. For example:

    autogoal ml fit autogoal/datasets/data/uci_cars/car.data --iterations 3 --pop-size 5 --format csv

![](autogoal_cli_ml_fit.svg)

Once a model has been trained, you can use it to predict another set. If you are predicting in training data, don't forget to use `--ignore-cols` to ignore the target column.

    autogoal ml predict autogoal/datasets/data/uci_cars/car.data --ignore-cols -1 --format csv

![](autogoal_cli_ml_predict.svg)

The results are stored in `output.csv`. You can change it with `--output <FILENAME>`.
