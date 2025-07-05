## Data Preprocess

To preprocess the data, you can use the `preprocess.py` script. It will convert the raw NL2SQL dataset into a format suitable for training.

```bash
bash examples/data_preprocess/nl2sql.sh
```

Then data will be saved in `data/nl2sql/` directory.
## Training
To train the model, you can use the `train.py` script. It will start the training process using the preprocessed data.

```bash
bash examples/train/nl2sql.sh
```