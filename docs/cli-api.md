# `autogoal`

🤩 Manage AutoGOAL directly from the CLI.

**Usage**:

```console
$ autogoal [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `contrib`: 🔍 Inspect contrib libraries and algorithms.
* `data`: 📚 Download, inspect, and generate training...
* `demo`: 🌟 Launch streamlit demo.
* `ml`: 🤖 Fit and predict with an AutoML model.

## `autogoal contrib`

🔍 Inspect contrib libraries and algorithms.

**Usage**:

```console
$ autogoal contrib [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: ⚙️ List all currently available contrib...

### `autogoal contrib list`

⚙️ List all currently available contrib algorithms.

**Usage**:

```console
$ autogoal contrib list [OPTIONS]
```

**Options**:

* `--verbose / --no-verbose`: [default: False]
* `--include TEXT`
* `--exclude TEXT`
* `--input TEXT`
* `--output TEXT`
* `--help`: Show this message and exit.

## `autogoal data`

📚 Download, inspect, and generate training data.

**Usage**:

```console
$ autogoal data [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `download`: ⏬ Download a dataset.
* `gen`: 🎲 Generate a random dataset.
* `list`: 🔍 List the available datasets.

### `autogoal data download`

⏬ Download a dataset.

Pass a name to directly download that dataset.
Otherwise, this command will show an interactive menu.

**Usage**:

```console
$ autogoal data download [OPTIONS] DATASETS...
```

**Arguments**:

* `DATASETS...`: Name of one or more specific datasets to download, or 'all'.  [required]

**Options**:

* `--help`: Show this message and exit.

### `autogoal data gen`

🎲 Generate a random dataset.

**Usage**:

```console
$ autogoal data gen [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `autogoal data list`

🔍 List the available datasets.

**Usage**:

```console
$ autogoal data list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `autogoal demo`

🌟 Launch streamlit demo.

**Usage**:

```console
$ autogoal demo [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## `autogoal ml`

🤖 Fit and predict with an AutoML model.

**Usage**:

```console
$ autogoal ml [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `fit`: 🏃 Train an AutoML instance on a dataset.
* `inspect`: 🔍 Inspect a trained AutoML model.
* `predict`: 🔮 Predict with a previously trained AutoML...

### `autogoal ml fit`

🏃 Train an AutoML instance on a dataset.

**Usage**:

```console
$ autogoal ml fit [OPTIONS] INPUT
```

**Arguments**:

* `INPUT`: [required]

**Options**:

* `--output PATH`: [default: automl.bin]
* `--target TEXT`
* `--ignore-cols INTEGER`: [default: ]
* `--evaluation-timeout INTEGER`: [default: 300]
* `--memory-limit INTEGER`: [default: 4294967296]
* `--search-timeout INTEGER`: [default: 3600]
* `--pop-size INTEGER`: [default: 20]
* `--iterations INTEGER`: [default: 100]
* `--format TEXT`
* `--help`: Show this message and exit.

### `autogoal ml inspect`

🔍 Inspect a trained AutoML model.

**Usage**:

```console
$ autogoal ml inspect [OPTIONS]
```

**Options**:

* `--model PATH`: [default: automl.bin]
* `--help`: Show this message and exit.

### `autogoal ml predict`

🔮 Predict with a previously trained AutoML instance.

**Usage**:

```console
$ autogoal ml predict [OPTIONS] INPUT
```

**Arguments**:

* `INPUT`: [required]

**Options**:

* `--output PATH`: [default: output.csv]
* `--model PATH`: [default: automl.bin]
* `--ignore-cols INTEGER`: [default: ]
* `--format TEXT`
* `--help`: Show this message and exit.

