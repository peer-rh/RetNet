from datasets import Dataset, load_dataset

SUPPORTED_DATASETS = ['rp-books']

def load_data(ds_name):
    assert ds_name in SUPPORTED_DATASETS 
    ds = _download_dataset(ds_name)
    # TODO: add preprocessing split into parts and tokenization
    return ds

def _download_dataset(ds_name):
    if ds_name == 'rp-books':
        ds = load_dataset("togethercomputer/RedPajama-Data-1T", subset='books')
        ds = ds.filter(lambda x: x["meta"]["language"] == "en")
        ds = ds.map(lambda x: x['text'])
        return ds
