# pyGAM datasets

These CSVs are copied byte for byte from pyGAM v0.12.0 (`pygam/datasets/*.csv` at tag
`v0.12.0` of https://github.com/dswah/pyGAM). That is the version the Python Contracts
bench venv installs (`bench/pygam_compare/requirements.txt`). pyGAM is licensed under the
Apache License 2.0, whose text, including its copyright line ("Copyright 2019 pyGAM
authors"), is in `LICENSE-pyGAM` beside this file.

`default.csv` is stored gzipped as `default.csv.gz`: its 10000 rows and header are one line
over the repository's 10k-line limit on tracked files (#780). `pg_helpers.dataset_dir()`
decompresses it, and links the others, into the directory the loaders read.

They are vendored because the pygam wheel ships the loaders in
`pygam.datasets.load_datasets` without the CSVs. `../conftest.py` points the loaders at
`dataset_dir()` so the translation suite reads the same bytes pyGAM's own tests read.

Only the datasets the suite loads are here: `mcycle`, `coal`, `wage`, `trees`, `default`,
`hepatitis_A_bulgaria` and `chicago`.
