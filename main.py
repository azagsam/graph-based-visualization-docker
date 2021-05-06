import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

from utils.helpers import load_sentences_from_file, load_sentences_from_AutoSentiNews, \
    load_sentences_from_cro_comments, scale_centrality_scores, \
    get_context_for_sentences

from utils.encoders import SentenceBERT

from utils.datasets import get_candas_doc, get_candas_doc_metadata, get_uploaded_example, get_generic_translations

import numpy as np

import base64
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import os
import uuid
import datetime

# stored values and metavariables
stored_values = {}
active_sessions = {}

SBERT_PATH = f'{os.path.abspath(os.getcwd())}/data/encoders/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'

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

    # DEBUG active sessions
    print('Number of active sessions:', len(active_sessions), 'Number of stored values:', len(stored_values))
    print('Active sessions:', active_sessions.keys())
    print('Stored values:', stored_values.keys())
    print('Autorefresh on!')

    return dbc.Container(
        fluid=True,
        children=[
            html.Div(id='page-3-content'),
            html.H1(f'Multilingual Text Exploration: {session_id}'),
            dcc.Store(data=session_id, id='session-id'),
            html.Hr(),

            dbc.Row([

                dbc.Nav(
                    [
                        dbc.NavLink("Active", active=True, href="/"),
                        dbc.NavLink("Instructions", href="/instructions"),

                    ]
                )

            ]),

            dbc.Row([

                dbc.Col(
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
                        dcc.Loading(id="loading-1",
                                    children=[html.Div(id="loading-output-1")],
                                    type="circle",
                                    style={'margin-top': '15px'}
                                    ),

                    ], body=True, style={'height': '220px'})),

                dbc.Col(
                    dbc.Card([

                        html.H2(html.Strong('Upload your data')),

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
                                'Drag and Drop or ',
                                html.A('Select File')
                            ]),
                            style={
                                # 'width': '40%',
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

                    ], body=True, style={'height': '220px'})),

                dbc.Col(
                    dbc.Card([

                        html.H2(html.Strong('Reload graph')),
                        html.H6('Select experiment:'),
                        dcc.Dropdown(id="reload-graph-dropdown",
                                     options=[
                                         # {"label": "0", "value": 0},
                                         # {"label": "1", "value": 1},
                                         # {"label": "2", "value": 2}
                                     ],
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

                    ], body=True, style={'height': '220px'})),

            ], style={'margin': '30px'}),

            dbc.Row([

                dbc.Col([dbc.Card(dcc.Graph(id='main-fig'))], width=10),

                dbc.Col(dbc.Card([

                    html.H3(html.Strong('Adjust graph')),

                    html.H6('Enter keyword (optional):'),
                    dcc.Input(id="keyword-input",
                              type="text",
                              placeholder="Enter keyword",
                              # value='None',
                              # style={'width': "20%"},
                              debounce=True),

                    html.H6('Limit number of sentences (optional):', style={'margin-top': '15px'}),
                    dcc.Input(id="num_of_sentences-input",
                              type="number",
                              placeholder="Enter num of sentences",
                              # value=-1,
                              # style={'width': "20%"},
                              debounce=True),

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

                ], body=True), width=2),

            ], style={'margin': '30px'}),

            dbc.Row([

            ], style={'margin': '30px'})

        ])


# app.layout = serve_layout

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


index_page = html.Div([
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.Link('Go to Page 3', href='/page-3'),
])

page_1_layout = html.Div([
    html.H1('Page 1'),
    dcc.Dropdown(
        id='page-1-dropdown',
        options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
        value='LA'
    ),
    html.Div(id='page-1-content'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
])

@app.callback(dash.dependencies.Output('page-1-content', 'children'),
              [dash.dependencies.Input('page-1-dropdown', 'value')])
def page_1_dropdown(value):
    return 'You have selected "{}"'.format(value)


page_2_layout = html.Div([
    html.H1('Page 2'),
    dcc.RadioItems(
        id='page-2-radios',
        options=[{'label': i, 'value': i} for i in ['Orange', 'Blue', 'Red']],
        value='Orange'
    ),
    html.Div(id='page-2-content'),
    html.Br(),
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

@app.callback(dash.dependencies.Output('page-2-content', 'children'),
              [dash.dependencies.Input('page-2-radios', 'value')])
def page_2_radios(value):
    return 'You have selected "{}"'.format(value)


# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return serve_layout()
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

@app.callback(
    Output('main-fig', 'figure'),  # update figure
    Output('reload-graph-dropdown', 'options'),  # update options of graph reload
    Output('reload-graph-dropdown', 'value'),  # to clear reload dropdown
    Output('loading-output-1', 'children'),  # message to display instead of the loading button
    Output('select_dataset-dropdown', 'value'),  # to clear demo datasets dropdown

    Input('generate-graph-button', 'n_clicks'),
    Input('reload-graph-button', 'n_clicks'),
    Input('keyword-input', 'value'),
    Input('num_of_sentences-input', 'value'),
    Input('slider_nodes', 'value'),
    Input('slider_edges', 'value'),
    Input('upload-data', 'contents'),

    State('reload-graph-dropdown', 'options'),
    State('reload-graph-dropdown', 'value'),
    State('select_dataset-dropdown', 'value'),
    State('select_language', 'value'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    State('session-id', 'data')
)
def update_graph(
        # inputs
        submit,
        reload,
        keyword,
        num_of_sentences,
        slider_nodes,
        slider_edges,
        list_of_contents,

        # states
        experiments,
        reload_graph_value,
        dataset,
        langid,
        list_of_names,
        list_of_dates,
        session_id
):

    # save session in stored_values if it does not exists yet
    if session_id not in stored_values.keys():
        stored_values[session_id] = {}

    # start building example ID
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
        prop_ids = ['keyword-input.value', 'slider_nodes.value', 'slider_edges.value']

        if ctx_msg['triggered'][0]['prop_id'] == 'upload-data.contents':
            print(list_of_names)
            example_id = f'uploaded_example_{list_of_names[0]}'

        elif ctx_msg['triggered'][0]['prop_id'] in prop_ids:
            assert 'current_example' in stored_values[session_id].keys()
            example_id = stored_values[session_id]['current_example']

    # check if experiment has run already
    if reload_graph_value:
        print(ctx_msg['states']['reload-graph-dropdown.value'])
        example_id = reload_graph_value

    # create new example id - refers to dropdown
    if not example_id:
        # check if num of sentences was changed for current example
        if num_of_sentences:
            example_id = '_'.join(stored_values[session_id]['current_example'].split('_')[:-1])  # remove num_of_sent from current example
            example_id = example_id + f'_{num_of_sentences}'  # add num_of_sents info

        # create example ID for Candas settings
        elif 'Candas' in dataset:
            dataset_name, candas_keyword = dataset.split(':')
            example_id = f'{dataset_name}_{candas_keyword}_{num_of_sentences}'

        # other: upload example, general translations etc
        else:
            example_id = f'{dataset}_{num_of_sentences}'

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
        elif 'uploaded_example' in example_id:  # todo add csv feature
            content_type, content_string = list_of_contents[0].split(',')
            decoded = base64.b64decode(content_string)
            decoded = decoded.decode('utf-8').replace('\n', ' ')
            sentences = get_uploaded_example(decoded, langid)
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
                if idx % 20 == 0:
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

        # calculate pagerank
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph, max_iter=500)  # number of cycles to converge

        # FEATURE: select the number of sentences
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

    # A) build figure without classes or clustering
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

    # B) show with classes (now works only for translations)
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

    # C) split candas based on metadata
    if 'metadata' in example_id:
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

    # build final figure with previously defined traces
    fig = go.Figure(data=fig_data,
                    layout=go.Layout(
                        title=f'<b>Experiment ID: "{example_id.split("_")[0]}", Number of sentences: {len(sentences)}</b>',
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

    # save experiment setup
    current_experiments = [e['label'] for e in experiments]
    if example_id not in current_experiments:
        exp_option = {'label': f'{example_id}', 'value': f'{example_id}'}
        experiments.append(exp_option)

    return fig, experiments, '', '', ''


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True, dev_tools_hot_reload=False)  # Turn off reloader if inside Jupyter
