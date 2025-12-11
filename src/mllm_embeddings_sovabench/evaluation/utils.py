import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import nltk

ANTONYMS = {'entering vehicle': 'exiting vehicle',
            'loading vehicle': 'unloading vehicle',
            'opening trunk': 'closing trunk',
            'opening vehicle door': 'closing vehicle door',
            'starting': 'stopping',
            'turning left': 'turning right'}

TWO_CLASS = {'vehicle': ['driving forward', 'reversing', 'starting', 'stopping', 'turning left', 'turning right'],
             'human-vehicle': ['entering vehicle', 'exiting vehicle', 'loading vehicle', 'unloading vehicle', 'opening trunk', 'closing trunk', 'opening vehicle door', 'closing vehicle door']
}


def load_embeddings(folder, embedder = 'all-MiniLM-L6-v2', multi_sentence=True):
    embeddings = dict()
    if folder[-5:] == '.json':
        with open(folder, 'r') as f:
            results = json.load(f)
        
        embedder = SentenceTransformer(embedder, trust_remote_code=True)
        for k, v in tqdm(results.items()):
            if multi_sentence:
                image_list = [t2 for t1 in v.split('\n') if t1 for t2 in nltk.sent_tokenize(t1)]
                if not image_list:
                    image_list = ['']
                emb = embedder.encode(image_list, normalize_embeddings=True)
            else:
                emb = embedder.encode([v], normalize_embeddings=True)
            embeddings[int(k)] = emb

    else:
        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            index = int(file.split('.')[0])
            emb = np.load(path)
            embeddings[index] = emb / np.linalg.norm(emb)
    return embeddings


def _map(df, embeddings, agg):
    sim = np.zeros((len(df['index']), len(df['index'])))
    for i, k1 in tqdm(enumerate(df['index'])):
        for j, k2 in enumerate(df['index']):
            cos = np.inner(embeddings[k1], embeddings[k2])
            
            if agg == 'max':
                sim[i, j] = cos.max()
            if agg == 'mean':
                sim[i, j] = cos.mean()
            if agg == 'median':
                sim[i, j] = np.median(cos)
    
    order = np.argsort(-sim, axis=1)

    APs = []
    for idx, row in df.iterrows():
        if row['subdim'] == 'distractor':
            continue

        i = df.index.get_loc(idx)
        
        # Exclude self
        ranked = order[i][order[i] != i]
        true_label = row['subdim']
        relevant = (df.iloc[ranked]['subdim'] == true_label).to_numpy().astype(int)

        # Compute precision at each relevant position
        cum_relevant = np.cumsum(relevant)
        precision_at_k = cum_relevant / (np.arange(len(relevant)) + 1)

        if np.sum(relevant) > 0:
            AP = np.sum(precision_at_k * relevant) / np.sum(relevant)
        else:
            AP = 0.0
        APs.append(AP)

    return np.mean(APs)


def mean_average_precision(embeddings, annotation_path, method='default', agg='max'):
    df = pd.read_csv(annotation_path, sep='\t')

    if method == 'binary':
        ANTONYMS['driving forward'] = 'reversing'
        
        mAPs = dict()
        for k, v in ANTONYMS.items():
            df_aux = df.loc[df['subdim'].isin([k, v])]
            mAPs[k] = _map(df_aux, embeddings, agg).item()
        mAPs['all'] = np.array(list(mAPs.values())).mean().item()
        return mAPs
    
    elif method == 'easy':
        mAPs = dict()
        for k, v in ANTONYMS.items():
            df.loc[df['subdim'] == v, 'subdim'] = k
            
        return _map(df, embeddings, agg)
    
    elif method == 'twoclass':
        mAPs = dict()
        for k, v in TWO_CLASS.items():
            for act in v:
                df.loc[df['subdim'] == act, 'subdim'] = k
            
        return _map(df, embeddings, agg)

    else:
        return _map(df, embeddings, agg)