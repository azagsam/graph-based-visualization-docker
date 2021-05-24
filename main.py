import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

from utils.helpers import load_sentences_from_file, load_sentences_from_AutoSentiNews, \
    load_sentences_from_cro_comments, scale_centrality_scores, \
    get_context_for_sentences, get_context_for_sentences

from utils.encoders import SentenceBERT

from utils.datasets import get_candas_doc, \
    get_candas_doc_metadata, \
    get_uploaded_example_txt, \
    get_uploaded_example_csv, \
    get_generic_translations

import numpy as np
import nltk

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import base64
import os
import io
import uuid
import datetime
import re

# stored values and metavariables
stored_values = {}
active_sessions = {}

SBERT_PATH = f'{os.path.abspath(os.getcwd())}/data/encoders/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'
NLTK_PATH = f'{os.path.abspath(os.getcwd())}/data/nltk_resources'
if NLTK_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_PATH)

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

# select and start encoder
encoder = encoders['SentenceBERT'](model_dir=SBERT_PATH)

# app layout
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__,
                prevent_initial_callbacks=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)


def serve_layout():
    # create session id
    session_id = str(uuid.uuid4())

    # check if there are any sessions older than two days and remove them
    active_sessions[session_id] = datetime.datetime.now()
    old_keys = []
    for key, value_datetime in active_sessions.items():
        delta = datetime.datetime.now() - value_datetime
        if delta.days > 2:
            stored_values.pop(key, None)
            print(f'Session {key} removed!', '\n')
            old_keys.append(key)
    # remove old keys from active sessions
    for key in old_keys:
        active_sessions.pop(key, None)

    # DEBUG session storing
    print('Number of active sessions:', len(active_sessions), 'Number of stored values:', len(stored_values))
    print('Active sessions:', active_sessions.keys())
    print('Stored values:', stored_values.keys())
    print('Autorefresh on!')

    return dbc.Container(
        fluid=True,
        children=[
            dcc.Store(data=session_id, id='session-id'),

            # Navigation buttons
            dbc.Row([

                dbc.Nav(
                    [
                        dbc.NavLink("Home", active=True, href="/"),
                        dbc.NavLink("Instructions", href="/instructions"),
                        dcc.Loading(id="loading-1",
                                    children=[html.Div(id="loading-output-1", style={'margin-left': '150px'})],
                                    type="default",
                                    ),

                    ], style={'margin-left': '30px'}
                )

            ], style={'margin': '15px'}),

            # Data import
            dbc.Row([

                # Demo datasets
                dbc.Col([
                    dbc.Card([

                        # html.H2(html.Strong('Configure data')),

                        html.H2(html.Strong('Demo datasets')),

                        html.H6('Select dataset:'),
                        dcc.Dropdown(id="select_dataset-dropdown",
                                     # Candas datasets values are structured as "name:keyword" pairs
                                     options=[
                                         {"label": "Generic translations", "value": "Generic translations"},

                                         {"label": "Candas (teorija)",
                                          "value": "Candas (teorija):teorija"},

                                         {"label": "Candas (globok)",
                                          "value": "Candas (globok):globok"},

                                         {"label": "Candas (teorija with metadata)",
                                          "value": "Candas (teorija with metadata):teorija"},

                                         {"label": "Candas (globok with metadata)",
                                          "value": "Candas (globok with metadata):teorija"},

                                         {"label": "Candas (globok with metadata, cluster bias)",
                                          "value": "Candas (globok with metadata, cluster bias):globok"},

                                         {"label": "Candas (globok with metadata, cluster source)",
                                          "value": "Candas (globok with metadata, cluster source):globok"},

                                         {"label": "Candas (kriza with metadata, cluster bias)",
                                          "value": "Candas (kriza with metadata, cluster bias):kriza"},

                                         {"label": "Candas (kriza with metadata, cluster source)",
                                          "value": "Candas (kriza with metadata, cluster source):kriza"},

                                     ],
                                     multi=False,
                                     placeholder="Select a dataset",
                                     # value='Candas',
                                     # style={'width': "40%"}
                                     ),
                        html.Div(id='select_dataset-dropdown-output-container'),

                        html.H6('Generate graph:', style={'margin-top': '15px'}),
                        html.Button('Generate graph',
                                    id='generate-graph-button',
                                    style={'background-color': 'lightskyblue'},
                                    n_clicks=0,
                                    ),

                    ], body=True, style={'height': '220px'}),
                    dbc.Card([

                        html.H2(html.Strong('Reload graph')),
                        html.H6('Select experiment:'),
                        dcc.Dropdown(id="reload-graph-dropdown",
                                     options=[],
                                     multi=False,
                                     placeholder="Select experiment",
                                     # value=0,
                                     # style={'width': "40%"}
                                     ),
                        html.H6('Reload graph:', style={'margin-top': '15px'}),
                        html.Button('Reload graph',
                                    id='reload-graph-button',
                                    style={'background-color': 'lightskyblue'},
                                    n_clicks=0),

                    ], body=True, style={'height': '220px', 'margin-top': '30px'}),

                ], width=3),

                # Upload your data
                dbc.Col([
                    dbc.Card([

                        html.H2(html.Strong('Upload data')),

                        html.H6('Select language:'),
                        dcc.Dropdown(id="select_language",
                                     options=[
                                         {"label": "Slovene", "value": 'slovene'},
                                         {"label": "English", "value": 'english'},
                                         {"label": "German", "value": 'german'},
                                     ],
                                     multi=False,
                                     placeholder="Select language",
                                     value='slovene',
                                     # style={'width': "40%"}
                                     ),
                        html.Div(id='select_language-output-container'),

                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                html.Button('Upload file',
                                            style={'background-color': 'lightskyblue'}, )
                            ]),
                            style={
                                'margin-top': '10px',
                            },
                            disabled=False,
                            # Allow multiple files to be uploaded
                            multiple=True
                        ),
                        html.Div(id='output-data-upload'),

                        dbc.Row([

                            dbc.Col([
                                html.H6('Select grouping type:', style={'margin-top': '15px'}),
                                dcc.RadioItems(
                                    id='radioitems',
                                    options=[
                                        {'label': ' No groups', 'value': 'None'},
                                        {'label': ' Cluster', 'value': 'cluster'},
                                        {'label': ' Classes', 'value': 'classes'},
                                    ],
                                    labelStyle={'display': 'block',
                                                'margin': '7px',
                                                },
                                    style={
                                        'display': 'inline-block',
                                        'margin-left': '10px'},
                                    value='None'
                                ),
                            ]),

                            # dbc.Col([
                            #     html.H6('Enter number of clusters:',
                            #             style={'margin-top': '15px'}),
                            #     dcc.Input(id="num_of_clusters-input",
                            #               type="number",
                            #               disabled=True,
                            #               placeholder="Enter num of clusters",
                            #               # value=5,
                            #               # style={'width': "20%"},
                            #               # debounce=False
                            #               ),
                            # ]),

                        ]),

                        html.H6('Enter number of clusters:',
                                style={'margin-top': '9px'}),
                        dcc.Input(id="num_of_clusters-input",
                                  type="number",
                                  disabled=True,
                                  placeholder="Enter num of clusters",
                                  # value=5,
                                  # style={'width': "20%"},
                                  # debounce=False
                                  ),
                        html.H6('Generate graph:', style={'margin-top': '15px'}),
                        html.Button('Generate graph',
                                    id='generate-graph-upload',
                                    style={'background-color': 'lightskyblue'},
                                    n_clicks=0,
                                    ),
                        # dcc.Loading(id="loading-1",
                        #             children=[html.Div(id="loading-output-1", style={'margin': '100px'})],
                        #             type="circle",
                        #             style={'margin': '10px', 'font-size': '50x'}
                        #             ),

                    ], body=True, style={'height': '470px'}),
                ], width=3),

                # User selected sentences
                dbc.Col([
                    dbc.Card([
                        # html.H2(html.Strong('Multilingual Text Exploration')),
                        # html.H6(f'Session ID: {session_id}'),
                        html.H2(html.Strong('User selected sentences')),
                        html.H6('Selected sentences from the graph will appear here:'),
                        html.Div([], style={'height': '400px',
                                            "overflow": "scroll",
                                            'border-style': 'solid',
                                            'border-width': '1px'}, id='sentence-div'),

                        # html.H6(f'Download selected sentences:'),
                        html.Button("Download sentences",
                                    id="download_button",
                                    style={'background-color': 'lightskyblue',
                                           'margin-top': '15px',
                                           #'width': '40%',
                                           'text-align': 'center',
                                           }),
                        dcc.Download(id="download"),
                    ], body=True, style={'height': '470px'})

                ], width=6)

            ], style={'margin': '15px'}
            ),

            # Figure
            dbc.Row([

                dbc.Col([dbc.Card(
                    # dcc.Loading(dcc.Graph(id='main-fig')), body=True)], width=10, style={'margin-top': '15px'}),
                dcc.Graph(id='main-fig'), body=True)], width = 10, style = {'margin-top': '15px'}),


                dbc.Col(dbc.Card([

                    html.H3(html.Strong('Adjust graph')),

                    html.H6('Enter keyword (optional):'),
                    dcc.Input(id="keyword-input",
                              type="text",
                              placeholder="Enter keyword",
                              # value='None',
                              # style={'width': "20%"},
                              debounce=True),

                    # html.H6('Limit number of sentences (optional):', style={'margin-top': '15px'}),
                    # dcc.Input(id="num_of_sentences-input",
                    #           type="number",
                    #           placeholder="Enter num of sentences",
                    #           # value=-1,
                    #           # style={'width': "20%"},
                    #           debounce=True),

                    html.H6('Scale nodes size:', style={'margin-top': '15px'}),
                    dcc.Slider(id='slider_nodes',
                               min=0,
                               max=1,
                               step=0.001,
                               value=0.5,
                               marks={
                                   0: '0',
                                   # 0.25: '0.25',
                                   # 0.50: '0.50',
                                   # 0.75: '0.75',
                                   1: '1'
                               },
                               ),
                    html.Div(id='slider_nodes-output-container'),

                    html.H6('Select edges threshold:', style={'margin-top': '15px'}),
                    dcc.Slider(id='slider_edges',
                               min=0,
                               max=1,
                               step=0.001,
                               value=0.8,
                               marks={
                                   0: '0',
                                   # 0.25: '0.25',
                                   # 0.50: '0.50',
                                   # 0.75: '0.75',
                                   1: '1'
                               },
                               ),
                    html.Div(id='slider_edges-output-container'),

                    html.H6('Contextualize uploaded data:', style={'margin-top': '15px'}),
                    dcc.RadioItems(id='context',
                                   options=[
                                       {'label': ' No', 'value': 'no'},
                                       {'label': ' Yes', 'value': 'yes'},
                                   ],
                                   value='no',
                                   labelStyle={'display': 'block',
                                               'margin': '7px',
                                               },
                                   style={
                                       'display': 'inline-block',
                                       'margin-left': '10px'},
                                   ),

                    html.H6('Select context size:', style={'margin-top': '15px'}),
                    dcc.Input(id='context-size',
                              type="number",
                              value=1,
                              # style={'width': "20%"},
                              debounce=True)

                ], body=True), width=2, style={'margin-top': '15px'}),

            ], style={'margin': '15px'}),

        ])


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

instructions = dbc.Container(
    fluid=True,
    children=[
        html.H1(f'Multilingual Text Exploration', style={'margin': '15px'}),
        html.Hr(),
        dbc.Row([

            dbc.Nav(
                [
                    dbc.NavLink("Home", active=True, href="/"),
                    dbc.NavLink("Instructions", href="/instructions"),

                ], style={'margin-left': '30px'}
            )
        ]),

        dbc.Row([dcc.Markdown('''
    
## How it works
This is a multilingual visualization tool. It takes raw text, tokenizes it into sentences, determines the importance 
of each sentence according to the graph-based model, and presents it in a figure that can be interactively explored. 
The importance of each sentence is indicated by how large the node is. The connections between sentences show how 
well they correlate.

## Usage
Graphs are constructed in two steps:

1) In the first row, you have three cards, from which you can configure data: demo datasets, upload your data, and 
   reload graph. The "Demo datasets" card contain preloaded experiments that show many ways to construct the graph. 
   After you run an experiment, it is saved into the memory, and can be reloaded fast with the "Reload graph" card. 
   "Upload you data" enables you to import your own data. Before you make an upload, you need to specify the 
   language of your text. You may want to cluster your data, or you have a csv file that contains pre-defined classes 
   (in that case, a csv file must contain pre-tokenized sentences, and two columns with names "class" and "sentence"). 
   
   
2) In the second row, the graph figure is presented on the left. On the right, the 'Adjust graph' card contains 
   options to update the graph. If you enter a keyword, all nodes that contain this keyword, will become the largest 
   in the graph. You can limit the number of sentences if you have a large set. For purely visualization purposes, 
   you can scale nodes size; higher number will emphasize larger nodes and make smaller ones smaller. Edges 
   threshold is a single cut-off value that either shows or hides connections between sentences; a very high value 
   will connect only sentences that highly correspond in their meaning (in extreme cases only duplicates). You may 
   also want to contextualize your data. 
    
    ''')], style={'width': '60rem', 'margin': '30px'})

    ])


# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/instructions':
        return instructions
    else:
        return serve_layout()


# Disable number of clusters
@app.callback(Output('num_of_clusters-input', 'disabled'),
              Input('radioitems', 'value'),
              )
def disable_clusters(radioitems):
    if radioitems == 'cluster':
        return False
    else:
        return True


# Enable/disable upload button
@app.callback(Output('upload-data', 'disabled'),
              Output('upload-data', 'children'),
              Input('upload-data', 'filename'),
              Input('generate-graph-upload', 'n_clicks'),
              )
def disable_upload_button(filename, generate_button):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'] == 'upload-data.filename':
        output_text = html.Div([f'{filename[0]} uploaded'])
        return True, output_text
    else:
        output_text = html.Div([
            # 'Drag and Drop or ',
            html.Button('Upload file',
                        style={'background-color': 'lightskyblue'}, )
        ])
        return False, output_text


# save user selected sentences
@app.callback(
    Output('sentence-div', 'children'),
    Input('main-fig', 'clickData'),  # you can also monitor "hoverData"
    State('sentence-div', 'children')
)
def callback(selection, sentence_div):
    labels = ['hovertemplate', 'text']
    for label in labels:
        if label in selection['points'][0].keys():
            # build sentence
            sentence = selection['points'][0][label]
            # remove tags
            sentence = re.sub('<[^>]*>', ' ', sentence)
            # build entry
            entry = [
                {
                    'props': {
                        'children': sentence
                    },
                    'type': 'P',
                    'namespace': 'dash_html_components'
                }
            ]
            print(sentence)
            return entry + sentence_div
    # when no point was selected
    return sentence_div


# download selected sentences
@app.callback(Output("download", "data"),
              Input("download_button", "n_clicks"),
              State('sentence-div', 'children')
              )
def generate_csv(n_nlicks, sentence_div):
    # create a list of sentences
    sentences = [entry['props']['children'] for entry in sentence_div]
    df = pd.DataFrame({
        'sentences': sentences
    })
    return dcc.send_data_frame(df.to_csv, filename="selected_sentences.csv")


# main callback
@app.callback(
    Output('main-fig', 'figure'),  # update figure
    Output('reload-graph-dropdown', 'options'),  # update options of graph reload
    Output('reload-graph-dropdown', 'value'),  # to clear reload dropdown
    Output('loading-output-1', 'children'),  # message to display instead of the loading button
    # Output('upload-data', 'filename'),  # clear uploaded filenames
    Output('select_dataset-dropdown', 'value'),  # to clear demo datasets dropdown
    # Output('num_of_clusters-input', 'value'),  # to clear num of clusters

    Input('generate-graph-button', 'n_clicks'),
    Input('reload-graph-button', 'n_clicks'),
    Input('generate-graph-upload', 'n_clicks'),
    Input('keyword-input', 'value'),
    # Input('num_of_sentences-input', 'value'),
    Input('slider_nodes', 'value'),
    Input('slider_edges', 'value'),
    Input('context', 'value'),
    Input('context-size', 'value'),

    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('reload-graph-dropdown', 'options'),
    State('reload-graph-dropdown', 'value'),
    State('select_dataset-dropdown', 'value'),
    State('select_language', 'value'),
    State('session-id', 'data'),
    State('radioitems', 'value'),
    State('num_of_clusters-input', 'value'),
)
def update_graph(

        # INPUTS
        submit,
        reload,
        generate_graph_upload,
        keyword,
        # num_of_sentences,
        slider_nodes,
        slider_edges,
        contextualize,
        context_size,

        # STATES
        list_of_contents,
        list_of_names,
        experiments,
        reload_graph_value,
        preloaded_dataset,
        langid,
        session_id,
        radioitem_value,
        num_of_clusters
):

    # save session in stored_values if it does not exist yet
    if session_id not in stored_values.keys():
        stored_values[session_id] = {}
        stored_values[session_id]['uploaded_csv'] = {}

    # start building example ID
    num_of_sentences = None
    example_id = None

    # for debbuging
    ctx = dash.callback_context
    ctx_msg = {
        'states': ctx.states,
        'triggered': ctx.triggered,
        'inputs': ctx.inputs
    }
    # print(ctx_msg)

    # check for special triggers
    if ctx_msg['triggered']:
        # check if trigger is in the following list
        prop_ids = ['keyword-input.value',
                    'slider_nodes.value',
                    'slider_edges.value']
        if ctx_msg['triggered'][0]['prop_id'] in prop_ids:
            # current example must be stored in session
            assert 'current_example' in stored_values[session_id].keys()
            example_id = stored_values[session_id]['current_example']

    # check if user wants to reload graph
    if reload_graph_value:
        print(ctx_msg['states']['reload-graph-dropdown.value'])
        example_id = reload_graph_value

    # create new example id
    if not example_id:
        # create example ID if preloaded dataset is chosen
        if preloaded_dataset:
            if 'Candas' in preloaded_dataset:
                dataset_name, candas_keyword = preloaded_dataset.split(':')
                example_id = f'{dataset_name}_{candas_keyword}_{num_of_sentences}'
            else:
                example_id = f'{preloaded_dataset}_{num_of_sentences}'

        # check if num of sentences was changed for current example
        elif num_of_sentences:
            example_id = '_'.join(
                stored_values[session_id]['current_example'].split('_')[:-1])  # remove num_of_sent from current example
            example_id = example_id + f'_{num_of_sentences}'  # add num_of_sents info

        # catch upload data
        elif list_of_names:
            print(list_of_names)
            example_id = f'upload_{list_of_names[0]}_{radioitem_value}'

    # store example id as current example
    stored_values[session_id]['current_example'] = example_id
    print(example_id)

    # check if example with the same id has already run, else run it from scratch
    if not example_id in stored_values[session_id].keys():
        if 'Generic translations' in example_id:
            sentences = get_generic_translations()
        elif 'metadata' in example_id:
            candas_keyword = example_id.split('_')[1]
            sentences, metadata = get_candas_doc_metadata(candas_keyword)
        elif 'Candas' in example_id:
            candas_keyword = example_id.split('_')[1]
            sentences = get_candas_doc(candas_keyword)
        elif example_id.startswith('upload'):
            content_type, content_string = list_of_contents[0].split(',')
            decoded = base64.b64decode(content_string)
            if 'csv' in list_of_names[0]:
                # csv must contain both columns: class, sentence
                uploaded_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                stored_values[session_id]['uploaded_csv'][example_id] = uploaded_df
                sentences = uploaded_df['sentence'].tolist()
            else:
                decoded = decoded.decode('utf-8').replace('\n', ' ')
                sentences = get_uploaded_example_txt(decoded, langid)
        else:
            print('Example not found, Figure has not been generated!')
            return go.Figure()

        # calculate word embeddings
        embeddings = encoder.encode_sentences(sentences, batch_size=4)

        # wrap sentences with '<br>' tags for improved visualization
        wraped_sentences = []
        for sentence in sentences:
            wraped_sentence = []
            for idx, word in enumerate(sentence.split()):
                if idx % 20 == 0 and idx != 0:
                    wraped_sentence.append(f'<br>{word}')
                else:
                    wraped_sentence.append(word)
            wraped_sentence = ' '.join(wraped_sentence)
            wraped_sentences.append(wraped_sentence)
        sentences = wraped_sentences

        # add metadata to candas dataset
        if 'metadata' in example_id:
            sentences = [f'{meta} <br> {sent}' for sent, meta in zip(sentences, metadata)]

        # similarity matrix
        sim_mat = cosine_similarity(embeddings)
        np.fill_diagonal(sim_mat, 0)

        # rescale
        scaler = MinMaxScaler(feature_range=(0, 1))
        sim_mat = scaler.fit_transform(sim_mat.flatten().reshape(-1, 1)).reshape(len(embeddings), len(embeddings))
        np.fill_diagonal(sim_mat, 0)

        # todo feature: cluster data before pagerank
        # calculate pagerank
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph, max_iter=500)  # number of cycles to converge

        # FEATURE: select number of sentences
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

    # FEATURE: wrap sentences with neighbour sentences
    if contextualize == 'yes':
        sentences = get_context_for_sentences(sentences, n=context_size)

    # FEATURE: zoom nodes that contain entered keyword
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

    ### Start building figure

    # use color names from this list: https://www.w3.org/wiki/CSS/Properties/color/keywords
    colors = ['green', 'blue', 'yellow', 'orange', 'purple', 'red', 'olive', 'lime',
              'navy', 'teal', 'aqua']

    # 1) create edges
    weights = sim_mat
    np.fill_diagonal(weights, 0)
    G = nx.from_numpy_array(weights)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        if weights[edge[0], edge[1]] > slider_edges:
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
        mode='lines',
        name='correlated')

    # 2) create nodes
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # A) show with classes (now works only for translations)
    if 'Generic translations' in example_id:

        color_keys = {
            'green': 'slovene',
            'orange': 'german',
            'blue': 'english',
            'yellow': 'croatian'

        }
        c = ['green', 'blue', 'yellow', 'orange']
        groups = ['green'] * 14 + ['blue'] * 14 + ['yellow'] * 14 + ['orange'] * 14
        fig_df = pd.DataFrame({
            'x': node_x,
            'y': node_y,
            'groups': groups,
            'centrality': centrality_scores,
            'sentences': sentences
        })

        fig_data = [edge_trace]
        for col in c:
            d = fig_df[fig_df['groups'] == col]
            scatter_single = go.Scatter(
                mode='markers',
                hovertemplate=[sent + '<extra></extra>' for sent in d['sentences'].tolist()],
                hovertext='text',
                x=d['x'],
                y=d['y'],
                opacity=0.5,
                marker=dict(
                    color=col,
                    size=[s * 10 for s in d['centrality']],
                    line=dict(
                        # color='MediumPurple',
                        width=1
                    )
                ),
                showlegend=True,
                name=color_keys[col]
            )
            fig_data.append(scatter_single)

    # B) split candas based on metadata
    elif 'metadata' in example_id:
        # select class
        if 'bias' in example_id:
            idx = 3
        elif 'source' in example_id:
            idx = 0
        else:
            idx = 1  # sentiment

        # create data
        cluster_name = [exp.split('<br>')[idx].split(':')[1].strip() for exp in sentences]
        cluster_unique = set(cluster_name)
        unique_colors = colors[:len(cluster_unique)]
        color_map = {b: c for b, c in zip(cluster_unique, unique_colors)}
        cluster_map = {c: b for b, c in zip(cluster_unique, unique_colors)}
        groups = [color_map[b] for b in cluster_name]

        fig_df = pd.DataFrame({
            'x': node_x,
            'y': node_y,
            'groups': groups,
            'centrality': centrality_scores,
            'sentences': sentences,
            'cluster_name': cluster_name
        })

        fig_data = [edge_trace]
        for col in unique_colors:
            d = fig_df[fig_df['groups'] == col]
            scatter_single = go.Scatter(
                mode='markers',
                hovertemplate=[sent + '<extra></extra>' for sent in d['sentences'].tolist()],
                hovertext='text',
                x=d['x'],
                y=d['y'],
                opacity=0.5,
                marker=dict(
                    color=col,
                    size=[s * 10 for s in d['centrality']],
                    line=dict(
                        # color='MediumPurple',
                        width=1
                    )
                ),
                showlegend=True,
                name=cluster_map[col]
            )
            fig_data.append(scatter_single)

    # C) cluster uploaded data
    elif example_id.startswith('upload') and (example_id.endswith('cluster') or example_id.endswith('classes')):

        # cluster
        if example_id.endswith('cluster'):
            # expand current example id with cluster information
            example_id_cluster = f'_clusters_{num_of_clusters}'

            # save results in session id
            if example_id_cluster not in stored_values[session_id].keys():
                print('Clusters were not calculated yet!')
                km = KMeans(n_clusters=num_of_clusters).fit(embeddings)
                classes = km.predict(embeddings)
                stored_values[session_id][example_id_cluster] = classes
            else:
                print('Reloading calculated clsuters ...')
                classes = stored_values[session_id][example_id_cluster]

        # classes
        else:
            uploaded_df = stored_values[session_id]['uploaded_csv'][example_id]
            classes = uploaded_df['class'].tolist()

        fig_df = pd.DataFrame({
            'x': node_x,
            'y': node_y,
            'classes': classes,
            'centrality': centrality_scores,
            'sentences': sentences
        })
        num_of_unique = len(fig_df['classes'].unique())
        map_indices = {idx: cls for idx, cls in enumerate(fig_df['classes'].unique())}

        fig_data = [edge_trace]
        for col in range(num_of_unique):
            d = fig_df[fig_df['classes'] == map_indices[col]]
            scatter_single = go.Scatter(
                mode='markers',
                hovertemplate=[sent + '<extra></extra>' for sent in d['sentences'].tolist()],
                hovertext='text',
                x=d['x'],
                y=d['y'],
                opacity=0.5,
                marker=dict(
                    color=col,
                    size=[s * 10 for s in d['centrality']],
                    line=dict(
                        # color='MediumPurple',
                        width=1
                    )
                ),
                showlegend=True,
                name=f'{map_indices[col]}'
            )
            fig_data.append(scatter_single)

    # D) build figure without classes or clustering
    else:
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            name='sentences',
            hoverinfo='text',
            marker=dict(
                # showscale=True,
                # colorscale options
                # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='Greens',
                size=[s * 10 for s in centrality_scores],
                colorbar=dict(
                    thickness=15,
                    title='Score',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=1))

        # FEATURE: wrap a sentence with neighbour sentences (useful for summarization to gain context)
        context = sentences
        node_adjacencies = []
        node_text = []
        for node, weight in enumerate(centrality_scores):
            node_adjacencies.append(weight)
            node_text.append(context[node])

        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        fig_data = [edge_trace, node_trace]

    # build final figure with previously defined traces
    fig = go.Figure(data=fig_data,
                    layout=go.Layout(
                        title=f'<b>Experiment ID: "{example_id}", Number of sentences: {len(sentences)}</b>',
                        height=900,
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        uirevision=example_id,
                        xaxis={'visible': False},
                        yaxis={'visible': False},
                        legend=dict(
                            # yanchor="top",
                            y=0.99,
                            # xanchor="left",
                            x=0.01,
                            traceorder="reversed",
                            title_font_family="Times New Roman",
                            font=dict(
                                # family="Courier",
                                size=12,
                                # color="black"
                            ),
                            # bgcolor="LightSteelBlue",
                            bordercolor="Black",
                            borderwidth=2
                        )
                    )
                    )

    # fig.update_layout(clickmode='select')

    # save experiment setup
    current_experiments = [e['label'] for e in experiments]
    if example_id not in current_experiments:
        exp_option = {'label': f'{example_id}', 'value': f'{example_id}'}
        experiments.append(exp_option)

    return fig, experiments, '', '', '',


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True, port=8050)  # debug=True, dev_tools_hot_reload=False
