# Pyllama-RAG

A simple CLI interface written to help my dad produce test questions more effectively from past year papers. Implements a couple of alternative splitting methods for improved retrieval accuracy, as well as a fully local querying system using Ollama.

# Installation and configuration

1. Install [Ollama](https://ollama.com/) and ensure that the binary is added to `$PATH`. The default embedding model is `bge-m3` and the default language model is `phi3:14b-medium-4k-instruct-q4_0`. These may be changed in `settings.py`.

2. Configure a virtual environment for the project using `venv` or `conda`. Use python version >= 3.12, as newline characters are used in f-strings.

```console
conda create --prefix ./.conda python=3.12
```

3. Subsequently, install required dependencies using `pip`.

```console
pip install -r requirements.txt
```

4. Further query customisation may be performed under `settings.py`. For example, the default prompt template and number of sources for query may be customised there. Conditionals are used so that different splitting methods, model providers and models may be specified using the CLI.
