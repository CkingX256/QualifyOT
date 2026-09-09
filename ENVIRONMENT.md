# Computational environment

Python 3.10 or later is recommended for the public package. Install the core and test dependencies with:

```bash
python -m venv .venv
# activate the environment
python -m pip install --upgrade pip
pip install -e ".[test]"
```

The optional Transformer candidate requires:

```bash
pip install -e ".[test,transformer]"
```

A machine-readable environment snapshot can be generated with:

```bash
python scripts/capture_environment.py
```

Random seeds, contract hashes and deterministic split salts used by reported analyses are retained in the repository because they are part of the reproducibility record.
