import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

from utils.helpers import load_sentences_from_file, load_sentences_from_AutoSentiNews, \
    load_sentences_from_cro_comments, scale_centrality_scores, \
    get_context_for_sentences

import stanza
from utils.encoders import SentenceBERT

from utils.datasets import get_candas, get_parlamint, get_kas, get_uploaded_example

import numpy as np

import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
# from flask_caching import Cache
import os
import uuid

STANZA_DIR = f'{os.path.abspath(os.getcwd())}/data/stanza_resources'
SBERT_PATH = f'{os.path.abspath(os.getcwd())}/data/encoders/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
user_id = str(uuid.uuid4())

# Sentence encoders and dimensionality reduction methods
encoders = {
    'SentenceBERT': SentenceBERT,
    # 'CMLM': CMLM,
    # 'LaBSE': LaBSE,
    # 'LASER': LASER
}

reduction_methods = {
    'pca': PCA,
    # 'umap': umap.UMAP,
    # 't-sne': TSNE,
    'None': None
}

reduction_methods_params = {
    'pca': {'n_components': 2},
    'umap': {'n_neighbors': 5, 'random_state': 42},  # check neighbors parameter
    't-sne': {'n_components': 2, 'perplexity': 30, 'random_state': 42},
    'None': None
}

cluster_params = {
    'kmeans': {'n_clusters': 3, 'random_state': 0},
    'gaussian_mixture': {'n_components': 3, 'covariance_type': 'full'}
}

# import slovene tokenizer TODO: add langid and nltk tokenizer
nlp = stanza.Pipeline('sl', dir=STANZA_DIR, use_gpu=True, processors='tokenize')

# select and start encoder
encoder = encoders['SentenceBERT'](model_dir=SBERT_PATH)

# app layout
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(prevent_initial_callbacks=True)
# cache = Cache(app.server, config={
#     #'CACHE_TYPE': 'redis',
#     # Note that filesystem cache doesn't work on systems with ephemeral
#     # filesystems like Heroku.
#     'CACHE_TYPE': 'filesystem',
#     'CACHE_DIR': 'cache-directory',
#
#     # should be equal to maximum number of users on the app at a single time
#     # higher numbers will store more data in the filesystem / redis cache
#     'CACHE_THRESHOLD': 200
# })
# control layout: https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/

def serve_layout():
    session_id = str(uuid.uuid4())

    return html.Div([
    html.H1(f'Multilingual text exploration: {session_id}'),
    dcc.Store(data=session_id, id='session-id'),

html.H4('---'*100),
html.H2('1. Configure data'),

    html.H4('1.1 Select dataset:'),
    dcc.Dropdown(id="select_dataset",
                 options=[
                     {"label": "ParlaMint", "value": 'ParlaMint'},
                     {"label": "Candas", "value": 'Candas'},
                 ],
                 multi=False,
                 placeholder="Select a dataset",
                 # value='Candas',
                 style={'width': "40%"}
                 ),
    html.Div(id='select_dataset-output-container'),

    html.H4('Select example:'),
    dcc.Dropdown(id="select_rows",
                 options=[
                     {"label": "0", "value": 0},
                     {"label": "1", "value": 1},
                     {"label": "2", "value": 2}
                 ],
                 multi=False,
                 placeholder="Select a row",
                 # value=0,
                 style={'width': "40%"}
                 ),

    html.H4('Enter number of sentences (optional):'),
    dcc.Input(id="enter_num_of_sentences",
              type="number",
              placeholder="Enter num of sentences or leave empty to select all sentences",
              #value=-1,
              style={'width': "20%"},
              debounce=True),

html.H4('Run experiment:'),
    html.Button('Run experiment',
                id='submit-val',
                style={'width': "10%"},
                n_clicks=0),
dcc.Loading(id="loading-1", children=[html.Div(id="loading-output-1")], type="default"),

    html.H4('---'*100),
html.H4('1.2 Upload your data (txt file):'),

html.H4('Select language:'),
    dcc.Dropdown(id="select_language",
                 options=[
                     {"label": "Slovene", "value": 'slovene'},
                     {"label": "English", "value": 'english'},
                     {"label": "German", "value": 'german'},
                 ],
                 multi=False,
                 placeholder="Select language",
                 # value='Candas',
                 style={'width': "40%"}
                 ),
    html.Div(id='select_language-output-container'),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '40%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),

html.H4('---'*100),
    html.H4('1.3. Reload experiment:'),
    dcc.Dropdown(id="dropdown",
                 options=[
                     # {"label": "0", "value": 0},
                     # {"label": "1", "value": 1},
                     # {"label": "2", "value": 2}
                 ],
                 multi=False,
                 placeholder="Select experiment",
                 # value=0,
                 style={'width': "40%"}
                 ),
    html.Button('Reload experiment',
                id='reload-exp',
                style={'width': "10%"},
                n_clicks=0),

html.H4('---'*100),

html.H2('2. Explore results'),

    html.H4('Enter keyword (optional):'),
    dcc.Input(id="keyword",
              type="text",
              placeholder="Enter keyword",
              # value='None',
              style={'width': "20%"},
              debounce=True),

html.H4('Scale nodes size:'),
    dcc.Slider(id='slider_nodes',
               min=0,
               max=1,
               step=0.001,
               value=0.5,
               marks={
                   0: '0',
                   0.25: '0.25',
                   0.50: '0.50',
                   0.75: '0.75',
                   1: '1'
               },
               ),
    html.Div(id='slider_nodes-output-container'),

html.H4('Select edges threshold:'),
    dcc.Slider(id='slider_edges',
                               min=0,
                               max=1,
                               step=0.001,
                               value=0.8,
                               marks={
                                   0: '0',
                                   0.25: '0.25',
                                   0.50: '0.50',
                                   0.75: '0.75',
                                   1: '1'
                               },
                               ),
    html.Div(id='slider_edges-output-container'),

html.Div(dcc.Graph(id='main-fig')),

])

app.layout = serve_layout


# stored values and metavariables
stored_values = {}


@app.callback(
    Output('main-fig', 'figure'),
    Output('dropdown', 'options'),
    Output('dropdown', 'value'),
    Output("loading-output-1", "children"),

    Input('submit-val', 'n_clicks'),
    Input('reload-exp', 'n_clicks'),
    Input('keyword', 'value'),
    Input('slider_nodes', 'value'),
    Input('slider_edges', 'value'),
    Input('upload-data', 'contents'),

    State('dropdown', 'options'),
    State('dropdown', 'value'),
    State('select_dataset', 'value'),
    State('select_language', 'value'),
    State('select_rows', 'value'),
    State('enter_num_of_sentences', 'value'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    State('session-id', 'data')
)
def update_graph(
        # inputs
        submit,
        reload,
        keyword,
        slider_nodes,
        slider_edges,
        list_of_contents,

        # states
        experiments,
        dropdown_value,
        dataset,
        langid,
        row,
        num_of_sentences,

        list_of_names,
        list_of_dates,

        session_id
):
    # for debbuging
    ctx = dash.callback_context
    ctx_msg = {
        'states': ctx.states,
        'triggered': ctx.triggered,
        'inputs': ctx.inputs
    }
    # print(ctx_msg)

    # save session in stored_values if it not exists yet
    if session_id not in stored_values.keys():
        stored_values[session_id] = {}

    example_id = ''
    # check for special triggers
    if ctx_msg['triggered']:
        prop_ids = ['keyword.value', 'slider_nodes.value', 'slider_edges.value']

        if ctx_msg['triggered'][0]['prop_id'] == 'upload-data.contents':
            print(list_of_names)
            example_id = f'uploaded_example_{list_of_names[0]}'

        elif ctx_msg['triggered'][0]['prop_id'] in prop_ids:
            assert 'current_example' in stored_values[session_id].keys()
            example_id = stored_values[session_id]['current_example']

    # check if experiment has been run already
    if dropdown_value:
        print(ctx_msg['states']['dropdown.value'])
        example_id = dropdown_value

    # create new example id
    if not example_id:
        example_id = f'dataset:{dataset}_example:{row}_numOfSents:{num_of_sentences}'

    stored_values[session_id]['current_example'] = example_id

    print(example_id)

    # check if example with the same id has already run, else run it from scratch
    if not example_id in stored_values[session_id].keys():
        if 'Candas' in example_id:
            sentences = get_candas(nlp, row)
        elif 'ParlaMint' in example_id:
            sentences = get_parlamint(row)
        elif 'uploaded_example' in example_id:
            content_type, content_string = list_of_contents[0].split(',')
            decoded = base64.b64decode(content_string)
            decoded = decoded.decode('utf-8').replace('\n', ' ')
            sentences = get_uploaded_example(nlp, decoded, langid)
        else:
            return go.Figure()

        # calculate word embeddings
        embeddings = encoder.encode_sentences(sentences, batch_size=4)

        # wrap sentences with '<br>' tags for improved visualization
        wraped_sentences = []
        for sentence in sentences:
            wraped_sentence = []
            for idx, word in enumerate(sentence.split()):
                if idx % 20 == 0:
                    wraped_sentence.append(f'<br>{word}')
                else:
                    wraped_sentence.append(word)
            wraped_sentence = ' '.join(wraped_sentence)
            wraped_sentences.append(wraped_sentence)
        sentences = wraped_sentences

        # similarity matrix
        sim_mat = cosine_similarity(embeddings)
        np.fill_diagonal(sim_mat, 0)

        # rescale
        scaler = MinMaxScaler(feature_range=(0, 1))
        sim_mat = scaler.fit_transform(sim_mat.flatten().reshape(-1, 1)).reshape(len(embeddings), len(embeddings))
        np.fill_diagonal(sim_mat, 0)

        # calculate pagerank
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph, max_iter=500)  # number of cycles to converge

        # select the number of sentences
        if not num_of_sentences:
            score_list = [scores[sent_idx] for sent_idx in range(len(sentences))]
        else:
            ranked_sentences = [(scores[idx], idx, s, e) for idx, (s, e) in enumerate(zip(sentences, embeddings))]
            ranked_sentences.sort(key=lambda x: x[0], reverse=True)
            sentences = []
            score_list = []
            embeddings = []
            unique = []
            for score, idx, s, e in ranked_sentences[:num_of_sentences]:
                sentences.append(s)
                score_list.append(score)
                embeddings.append(e.tolist())
                unique.append(idx)

            # create a reduced sim matrix
            reduced_sim_mat = []
            for vec_id in unique:
                smaller_vec = sim_mat[vec_id][unique]
                reduced_sim_mat.append(smaller_vec)
            sim_mat = np.asarray(reduced_sim_mat)

        # reduce dimensionality
        rec_method_name = 'pca'
        reduction_method = reduction_methods['pca']
        print(rec_method_name)
        assert rec_method_name != 'None'
        rm = reduction_method(**reduction_methods_params[rec_method_name])
        pos = rm.fit_transform(embeddings)

        stored_values[session_id][example_id] = (sentences, embeddings, sim_mat, score_list, pos)
    else:
        sentences, embeddings, sim_mat, score_list, pos = stored_values[session_id][example_id]

    # FEATURE: zoom nodes that contain keyword
    if keyword:
        max_score = max(score_list)
        updated_score_list = []
        for sent, score in zip(sentences, score_list):
            if keyword in sent:
                updated_score_list.append(max_score)
            else:
                updated_score_list.append(score)
        score_list = updated_score_list

    # FEATURE: scale nodes for visualization purposes
    centrality_scores = np.array(score_list)
    if slider_nodes:
        centrality_scores = scale_centrality_scores(centrality_scores, q=slider_nodes)

    # create edges
    weights = sim_mat
    np.fill_diagonal(weights, 0)
    G = nx.from_numpy_array(weights)

    edge_x = []
    edge_y = []
    draw_line = slider_edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if weights[edge[0], edge[1]] > draw_line:
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1,
                  color='#888'),
        hoverinfo='none',
        mode='lines')

    # create nodes
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='Greens',
            # reversescale=True,
            color=[],
            size=[s * 10 for s in centrality_scores],
            colorbar=dict(
                thickness=15,
                title='Score',
                xanchor='left',
                titleside='right'
            ),
            line_width=1))

    # FEATURE: wrap a sentence with neighbour sentences
    context = sentences
    node_adjacencies = []
    node_text = []
    for node, weight in enumerate(centrality_scores):
        node_adjacencies.append(weight)
        node_text.append(context[node])

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # create a figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'<b>Experimet ID: "{example_id.replace("_", " ")}", Number of sentences: {len(sentences)}</b>',
                        height=800,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        uirevision=example_id
                        # this line does not reset zoom: https://community.plotly.com/t/preserving-ui-state-like-zoom-in-dcc-graph-with-uirevision-with-dash/15793
                        # annotations=[ dict(
                        #     #text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'>
                        #     # https://plotly.com/ipython-notebooks/network-graphs/</a>",
                        #     showarrow=False,
                        #     xref="paper", yref="paper",
                        #     x=0.005, y=-0.002 ) ],
                        # xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        # yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    ))

    # save experiment setup
    current_experiments = [e['label'] for e in experiments]
    if example_id not in current_experiments:
        exp_option = {'label': f'{example_id}', 'value': f'{example_id}'}
        experiments.append(exp_option)

    return fig, experiments, '', 'Experiment was loaded!'

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)  # Turn off reloader if inside Jupyter
