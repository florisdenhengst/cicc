import pandas as pd
import itertools

DATA_ROOT = 'multilingual_task_oriented_dialog_slotfilling'

LANGUAGES = ['en','es','th']
SPLITS = ['train', 'eval', 'test']
COLUMNS = ['utterance', 'labels']


def get_path(split, language, format):
    if language == 'th':
        return f"{DATA_ROOT}/{language}/{split}-{language}_TH.{format}"
    return f"{DATA_ROOT}/{language}/{split}-{language}.{format}"

def get_to_path(split, language, format):
    if split == 'eval':
        split = 'val'
    return f"{format}/{language}/{split}.{format}"

def read_data(split, language):
    path = get_path(split, language, 'tsv')
    data = pd.read_csv(
        path,
        sep='\t',
        header=None,
        names=['domain_intent', 'tokens', 'utterance', 'language', 'token_spans'])
    return data

def normalize_intents(data):
    data[['domain', 'labels']] = data['domain_intent'].str.split('/', n=1, expand=True)
    return data

def tsv_to_csv(split, language):
    data = read_data(split, language)
    data = normalize_intents(data)
    to_path = get_to_path(split, language, 'csv')
    data[COLUMNS].to_csv(to_path, index=False)

for (split, language) in itertools.product(SPLITS, LANGUAGES):
    tsv_to_csv(split, language)