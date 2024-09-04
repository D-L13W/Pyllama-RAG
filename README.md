# Pyllama-RAG

## Introduction

A simple CLI interface written to help my dad produce test questions more effectively from past year papers. Implements a couple of alternative splitting methods for improved retrieval accuracy, as well as a fully local querying system using Ollama by default.

## Installation and configuration

1. Install [Ollama](https://ollama.com/) and ensure that the binary is added to `$PATH`. This is the default provider. The default embedding model is `bge-m3` and the default language model is `phi3:14b-medium-4k-instruct-q4_0`. These may be changed in `defaults.py`.

2. Configure a virtual environment for the project using `venv` or `conda`. Use python version >= 3.12, as newline characters are used in f-strings.

```console
conda create --prefix ./.conda python=3.12
```

3. Subsequently, install required dependencies using `pip`.

```console
pip install -r requirements.txt
```

4. Most customisations may be performed using CLI flags for `refresh_db.py` and `query_data.py` (defined in `cli_flags.py`). The only variable which cannot be specified in this way is the prompt template (accessible under `defaults.py`) because it is unnecessarily long to include as a CLI argument. Pointing to a filepath is a valid alternative, but editing the template in `defaults.py` directly is more straightfoward than having an external file.

## Usage

- The two main scripts used in the CLI are `refresh_db.py` and `query_data.py`. The purposes of these files are self explanatory. These files may be run by python with the `-h` flag to print relevant CLI flags.

- The CLI flags are specified in `cli_flags.py`. A separate module was written because both of the two files above may share common flags.

- To add more splitting methods and model providers, `split_methods.py` and `model_providers.py` may be edited.

- Finally, a collation of default settings are provided in `defaults.py`.

The scripts are written such that alternative models, model providers and splitting methods may be easily specified via the CLI. It is also easy to add on more options for these, by importing necessary packages and modifying/adding methods under `model_providers.py` and `split_methods.py` respectively. The new methods can then be added to the CLI by editing the appropriate choice list variables in the respective files.
