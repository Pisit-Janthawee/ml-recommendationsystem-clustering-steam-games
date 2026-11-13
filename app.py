"""Steam game recommender, cluster explorer and thumbnail GAN - Gradio app.

The app builds its index from the Kaggle dataset on first start (about 1 minute) and caches it in ./cache.
No data is stored in the repository. Where the CSV files are looked for, in order:
  1. the folder in the environment variable STEAM_DATA_DIR
  2. ./data
  3. a download with kagglehub (needs Kaggle credentials: KAGGLE_USERNAME / KAGGLE_KEY)

Run:  pip install -r requirements.txt  ->  python app.py
"""
import html
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfTransformer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache' / 'index.pkl'
GAN_WEIGHTS = ROOT / 'models' / 'generator_ema.pt'
SEED = 42
TOP_K = 10
POOL = 100
BETA = 0.2                 # weight of review quality when re-ranking (chosen in the notebook)
HYBRID_WEIGHT = 0.5
N_CLUSTERS = 18
EXAMPLES = ['Stardew Valley', 'Portal 2', 'Cities: Skylines', 'DARK SOULS™ III', 'Counter-Strike: Global Offensive']


# ---------------------------------------------------------------- data
def find_data_dir():
    for folder in [os.environ.get('STEAM_DATA_DIR'), ROOT / 'data']:
        if folder and (Path(folder) / 'steam.csv').is_file():
            return Path(folder)
    import kagglehub
    return Path(kagglehub.dataset_download('nikdavis/steam-store-games'))


STEMMER, STOP = PorterStemmer(), set(ENGLISH_STOP_WORDS)
TOKEN = re.compile(r'[a-z][a-z0-9]+')
TAG_RE = re.compile(r'<[^>]+>')
URL_RE = re.compile(r'https?://\S+|www\.\S+|\S+\.(?:com|net|org)\S*')


def normalize_text(text):
    text = URL_RE.sub(' ', html.unescape(TAG_RE.sub(' ', str(text)))).lower()
    return ' '.join(STEMMER.stem(w) for w in TOKEN.findall(text) if w not in STOP)


def wilson(pos, n, z=1.96):
    n = np.asarray(n, dtype=float)
    p = np.divide(pos, n, out=np.zeros_like(n), where=n > 0)
    m = n.clip(min=1)
    return np.where(n > 0, (p + z ** 2 / (2 * m) - z * np.sqrt(p * (1 - p) / m + z ** 2 / (4 * m ** 2))) / (1 + z ** 2 / m), 0.0)


def build_index():
    data_dir = find_data_dir()
    store = pd.read_csv(data_dir / 'steam.csv')
    desc = pd.read_csv(data_dir / 'steam_description_data.csv').rename(columns={'steam_appid': 'appid'})
    tags = pd.read_csv(data_dir / 'steamspy_tag_data.csv')
    media = pd.read_csv(data_dir / 'steam_media_data.csv', usecols=['steam_appid', 'header_image']).rename(columns={'steam_appid': 'appid'})
    games = (store.merge(desc[['appid', 'short_description', 'about_the_game']], on='appid')
                  .merge(tags[['appid']], on='appid').merge(media, on='appid', how='left')
                  .drop_duplicates('appid').reset_index(drop=True))
    games['developer'] = games['developer'].fillna('unknown')
    games['ratings'] = games['positive_ratings'] + games['negative_ratings']
    games['wilson'] = wilson(games['positive_ratings'], games['ratings'])
    tag_df = tags.set_index('appid').loc[games['appid']]
    tag_names = tag_df.columns.tolist()

    text = (games['short_description'].fillna('') + ' ' + games['about_the_game'].fillna('')).map(normalize_text)
    tfidf = TfidfVectorizer(min_df=5, max_df=0.5, max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)
    X_text = tfidf.fit_transform(text).astype(np.float32)
    X_tags = normalize(TfidfTransformer(sublinear_tf=True).fit_transform(sparse.csr_matrix(tag_df.to_numpy(dtype=np.float32))))
    X = sparse.hstack([np.sqrt(HYBRID_WEIGHT) * X_tags, np.sqrt(1 - HYBRID_WEIGHT) * X_text]).tocsr().astype(np.float32)

    Z = normalize(TruncatedSVD(100, random_state=SEED).fit_transform(X_tags))
    games['cluster'] = KMeans(N_CLUSTERS, n_init=10, random_state=SEED).fit_predict(Z)
    vocab = np.array(tfidf.get_feature_names_out())
    dense_tags = X_tags.toarray()
    overall_words = np.asarray(X_text.mean(axis=0)).ravel()
    clusters = []
    for c in range(N_CLUSTERS):
        m = (games['cluster'] == c).to_numpy()
        top_tags = [tag_names[i] for i in np.argsort(-dense_tags[m].mean(axis=0))[:5]]
        words = vocab[np.argsort(-(np.asarray(X_text[m].mean(axis=0)).ravel() - overall_words))[:5]]
        clusters.append({'cluster': c, 'games': int(m.sum()), 'label': ', '.join(top_tags[:3]),
                         'top tags': ', '.join(top_tags), 'distinctive words': ', '.join(words)})
    keep = ['appid', 'name', 'developer', 'genres', 'wilson', 'ratings', 'cluster', 'header_image']
    return {'games': games[keep], 'X': X, 'clusters': pd.DataFrame(clusters)}


def load_index():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    index = build_index()
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_bytes(pickle.dumps(index))
    return index


INDEX = load_index()
GAMES, X, CLUSTERS = INDEX['games'], INDEX['X'], INDEX['clusters']
QUALITY = GAMES['wilson'].to_numpy()


# ---------------------------------------------------------------- recommender
def find_game(title):
    names = GAMES['name'].str.lower()
    exact = GAMES.index[names == title.strip().lower()]
    if len(exact):
        return GAMES.loc[exact, 'ratings'].idxmax()
    hits = GAMES[names.str.contains(re.escape(title.strip().lower()))]
    return hits['ratings'].idxmax() if len(hits) else None


def recommend(title, beta=BETA, k=TOP_K):
    q = find_game(title or '')
    if q is None:
        return pd.DataFrame({'message': [f'No Steam game matching "{title}"']}), []
    sim = linear_kernel(X[q], X).ravel().astype(np.float64)
    sim[q] = -1
    pool = np.argpartition(-sim, POOL)[:POOL]
    score = (1 - beta) * sim[pool] + beta * QUALITY[pool]
    picks = pool[np.argsort(-score)[:k]]
    recs = GAMES.loc[picks].copy()
    recs['match %'] = np.round(sim[picks] * 100, 1)
    table = recs[['name', 'match %', 'developer', 'genres', 'wilson']].rename(columns={'wilson': 'quality'}).round(3)
    gallery = [(u, f'{n} | {m:.1f} %') for u, n, m in zip(recs['header_image'], recs['name'], recs['match %']) if isinstance(u, str)]
    return table, gallery


# ---------------------------------------------------------------- clusters
CLUSTER_CHOICES = [f"{r.cluster}: {r.label} ({r.games} games)" for r in CLUSTERS.sort_values('games', ascending=False).itertuples()]


def explore(choice):
    c = int(str(choice).split(':')[0])
    info = CLUSTERS.loc[CLUSTERS['cluster'] == c].iloc[0]
    members = GAMES[(GAMES['cluster'] == c) & (GAMES['ratings'] >= 50)].sort_values('wilson', ascending=False).head(12)
    text = f"**Top tags** → {info['top tags']}\n\n**Distinctive words** → {info['distinctive words']}\n\n**Games** → {info['games']:,}"
    return text, [(u, n) for u, n in zip(members['header_image'], members['name']) if isinstance(u, str)]


# ---------------------------------------------------------------- GAN
GENERATOR = None
if GAN_WEIGHTS.exists():
    import torch
    import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self, z=128, f=64):
            super().__init__()
            self.f = f
            self.fc = nn.Sequential(nn.Linear(z, f * 8 * 4 * 8, bias=False), nn.BatchNorm1d(f * 8 * 4 * 8), nn.ReLU(True))
            def up(cin, cout):
                return [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(cin, cout, 3, 1, 1, bias=False),
                        nn.BatchNorm2d(cout), nn.ReLU(True)]
            self.net = nn.Sequential(*up(f * 8, f * 4), *up(f * 4, f * 2), *up(f * 2, f), *up(f, f // 2),
                                     nn.Conv2d(f // 2, 3, 3, 1, 1), nn.Tanh())

        def forward(self, z):
            return self.net(self.fc(z).view(z.size(0), self.f * 8, 4, 8))

    ckpt = torch.load(GAN_WEIGHTS, map_location='cpu')
    GENERATOR = Generator(ckpt['latent_dim'])
    GENERATOR.load_state_dict(ckpt['state_dict'])
    GENERATOR.eval()


def _to_images(batch):
    x = ((batch.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
    return [np.kron(im, np.ones((3, 3, 1), dtype=np.uint8)) for im in x]      # 3x nearest upscale for display


def generate(seed, n=8):
    if GENERATOR is None:
        return []
    import torch
    g = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        return _to_images(GENERATOR(torch.randn(int(n), 128, generator=g)))


def walk(seed_a, seed_b, steps=8):
    if GENERATOR is None:
        return []
    import torch
    za = torch.randn(1, 128, generator=torch.Generator().manual_seed(int(seed_a)))
    zb = torch.randn(1, 128, generator=torch.Generator().manual_seed(int(seed_b)))
    omega = torch.acos((za / za.norm() * zb / zb.norm()).sum())
    zs = torch.cat([(torch.sin((1 - t) * omega) * za + torch.sin(t * omega) * zb) / torch.sin(omega)
                    for t in torch.linspace(0, 1, int(steps))])
    with torch.no_grad():
        return _to_images(GENERATOR(zs))


# ---------------------------------------------------------------- UI
def build_app():
    import gradio as gr
    with gr.Blocks(title='Steam Game Recommendation System') as demo:
        gr.Markdown('# Steam Game Recommendation System\nContent-based recommendations (player tags + descriptions, '
                    're-ranked by review quality), clusters of the catalogue, and thumbnails invented by a GAN.')
        with gr.Tab('Recommend'):
            title = gr.Textbox(label='A Steam game you like', value='Stardew Valley')
            beta = gr.Slider(0, 0.5, value=BETA, step=0.05, label='Weight of review quality (0 = similarity only)')
            go = gr.Button('Recommend', variant='primary')
            table = gr.Dataframe(label='Similar games')
            cards = gr.Gallery(label='Header images', columns=5, height='auto')
            gr.Examples([[e] for e in EXAMPLES], inputs=title)
            go.click(recommend, [title, beta], [table, cards])
            title.submit(recommend, [title, beta], [table, cards])
        with gr.Tab('Explore clusters'):
            choice = gr.Dropdown(CLUSTER_CHOICES, value=CLUSTER_CHOICES[0], label='Cluster (by size)')
            info = gr.Markdown()
            shelf = gr.Gallery(label='Best-reviewed games in this cluster', columns=4, height='auto')
            choice.change(explore, choice, [info, shelf])
            demo.load(explore, choice, [info, shelf])
        with gr.Tab('Generate thumbnails (GAN)'):
            if GENERATOR is None:
                gr.Markdown('(!) `models/generator_ema.pt` not found: train it with `steam_thumbnail_gan.ipynb`.')
            seed = gr.Number(value=7, precision=0, label='Seed')
            gen = gr.Button('Generate 8 thumbnails', variant='primary')
            fakes = gr.Gallery(label='Generated thumbnails (64 x 128, shown 3x)', columns=4, height='auto')
            gen.click(generate, seed, fakes)
            with gr.Row():
                seed_a = gr.Number(value=1, precision=0, label='From seed')
                seed_b = gr.Number(value=2, precision=0, label='To seed')
            morph = gr.Button('Walk through the latent space')
            path = gr.Gallery(label='Interpolation', columns=8, height='auto')
            morph.click(walk, [seed_a, seed_b], path)
    return demo


if __name__ == '__main__':
    build_app().launch()
