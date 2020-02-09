# autozoom is not working properly
# dropdown selection is not working properly
# link heatmap with epoch
# restructure the web template
# animate on a play button

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly


import csv
import json
from plotly import tools


#GRAPH CODE ADDED
import plotly.graph_objs as go
import networkx as nx


################### START OF DASH APP ###################

app = dash.Dash(__name__)

#ice = cmocean_to_plotly(cmocean.cm.ice, max_len)
#py.iplot(colorscale_plot(colorscale=ice, title='Ice'))


@app.server.before_first_request
def load_demo_run_logs():
    print("I am first!!")
    global dataset_dict

    dataset_dict = {
        'terrorist': {
            'edges': "Data/Graph/TerrorAttack_edges.csv",
            'graph': "Data/Graph/TerrorAttack_graph.csv",
            'epochs': pd.read_csv('Data/TSNE/Terr_tsne_epochs.csv'),
            'X0': pd.read_csv('Data/Heatmap Dataset/Terr_X0.csv', header=None),
            'X1': pd.read_csv('Data/Heatmap Dataset/Terr_X1_Epochs.csv', index_col=[0]),
            'X2': pd.read_csv('Data/Heatmap Dataset/Terr_X2_Epochs.csv', index_col=[0])
        },
        'karate': {
            'edges': "Data/Graph/Karate_edges.csv",
            'graph': "Data/Graph/Karate_graph.csv",
            'epochs': pd.read_csv('Data/TSNE/Karate_tsne_epochs.csv'),
            'X0': pd.read_csv('Data/Heatmap Dataset/Karate_X0.csv', header=None),
            'X1': pd.read_csv('Data/Heatmap Dataset/Karate_X1_Epochs.csv', index_col=[0]),
            'X2': pd.read_csv('Data/Heatmap Dataset/Karate_X2_Epochs.csv', index_col=[0])
        }
    }

server = app.server

#Heatmap data
def process_X(df, epoch=-1):
    if epoch == -1:
        y = df.index.values + 1
        x = np.array(range(len(df.columns))) + 1
    
        values = df.values
    else:
        df_subset = df.loc[df["Epochs"] == epoch, df.columns != 'Epochs'].reset_index(drop=True)
        
        x = np.array(range(len(df_subset.columns))) + 1
        y = df_subset.index.values + 1

        values = df_subset.values

    return x, y, values    

#Heatmap layout and data
app.layout = html.Div([
    
    #Heading and Dropdown
    html.Div([
        html.Div([
            html.H2('Graph Convolution Network Model Analyser')
        ]),
        html.Div([
            html.H5('By Mridula Gupta and Sarmishta Burujupalli')
        ]),

        #Left Filter Div 
        html.Div([
            
            
            #Slider and Filter Div
            html.Div([
                #Filter - Div 1
                html.Div([
                    html.Div([
                        html.P('Dataset:'),
                    ],style={'width': '15%', 'display': 'inline-block'}),
                    
                    html.Div([
                        dcc.Dropdown(
                            id='dropdown-dataset',
                            options=[
                                {'label': 'Karate', 'value': 'karate'},
                                {'label': 'Terrorist', 'value': 'terrorist'}
                            ],
                            placeholder="Select Dataset",
                            searchable=True,
                            value="karate",
                            clearable=False
                    )],style={'width': '85%', 'display': 'inline-block'})
                    ],style={'textAlign':'left'}),
                    #'textAlign': 'center'

                #SliderButton - Div 3
                html.Div([
                        html.Div([
                            html.P('Epochs:'),
                        ],style={'width': '15%', 'display': 'inline-block'}),
                        
                        html.Div([
                        dcc.Slider(
                            id='epochs-slider',
                            min=0,
                            max=200,
                            value=0,
                            step=10,
                            marks={str(epochs):str(epochs) for epochs in range(0, 210, 10)}
                        ),
                        dcc.Interval(
                            id='auto-stepper',
                            interval=1*1000, # in milliseconds
                            n_intervals=0
                        )
                    ],style={'width': '85%', 'display': 'inline-block'}),
                    
                    ],style={'textAlign':'left'}),
            ],style={'width': '83%', 'display': 'inline-block'}),

            #PLay-Pause Button - Div 2
            html.Div([
                dcc.Tabs(
                    id="tabs", 
                    value='play', 
                    vertical='vertical',
                    style= {'font-size': '15px','background-color':'#3366ff','height':'30px'},
                    children=[
                    dcc.Tab(label='Play', value='play'),
                    dcc.Tab(label='Pause', value='pause'),
                    ]
                )],style={'width': '17%','textAlign': 'right','display': 'inline-block','vertical-align': 'top'}) #,'borderLeft': 'thin lightgrey solid'
        ],style={'width': '45.5%', 'display': 'inline-block','height':'125px','margin-left':'2%','border-style': 'solid','border-color':'#f2f2f2','padding':'0.5%','border-width': 'thin'}), #'margin-right':'5%' 
        
        #Center  Div 
        html.Div([
            html.Div([
                 html.P('Dataset Information: ')
                ]),
            html.Div([
                    html.Div(id='num_nodes'),
                    html.Div([
                        html.P('Nodes')
                    ])
                ],style={'display': 'inline-block','width':'30%'}),
            html.Div([
                    html.Div(id='num_edges'),
                    html.Div([
                        html.P('Edges')
                    ])
                ],style={'display': 'inline-block','width':'30%'}),
            html.Div([
                    html.Div(id='num_features'),
                    html.Div([
                        html.P('Features')
                    ])
                ],style={'display': 'inline-block','width':'30%'})

        ],style={'width': '22%', 'margin-left':'1%' , 'display': 'inline-block','height':'125px','vertical-align': 'top','border-style': 'solid','border-color':'#f2f2f2','padding':'0.5%','border-width': 'thin'}),

        #Right Filter Div 
        html.Div([

            html.Div([
                html.Div([
                        html.P("Change #Epochs:"),
                    ],style={'display': 'inline-block','margin-right':'2%','width':'60%','textAlign':'left'}),

                html.Div([
                        html.Button('-',  id='EpochDown', n_clicks_timestamp='0')
                    ],style={'display': 'inline-block','width':'7%'}),
            
                html.Div([
                        html.P("200"),
                    ],style={'display': 'inline-block','margin-left':'3%','margin-right':'3%','width':'9%'}),
                
                html.Div([
                        html.Button('+',  id='EpochUp', n_clicks_timestamp='0'),
                    ],style={'display': 'inline-block','width':'5%'})
            ]),

            html.Div([
                html.Div([
                        html.P("Change #Hidden Layers:"),
                    ],style={'display': 'inline-block','margin-right':'2%','width':'60%','textAlign':'left'}),

                html.Div([
                        html.Button('-',  id='FeatureDown', n_clicks_timestamp='0')
                    ],style={'display': 'inline-block','width':'6%'}),
            
                html.Div([
                        html.P("16"),
                    ],style={'display': 'inline-block','margin-left':'3%','margin-right':'3%','width':'9%'}),
                
                html.Div([
                        html.Button('+',  id='FeatureUp', n_clicks_timestamp='0'),
                    ],style={'display': 'inline-block','width':'5%'})
            ]),

            html.Div([
                html.Button('Update Model [WIP]',  
                id='btn-1', 
                n_clicks_timestamp='0',
                style= {'font-size': '15px','color':'black','background-color':'#f2f2f2'}
                #pad = {'r': 10, 't': 10},
                ),
                ],style={'margin':'auto'})

            ],style={'width': '21.5%', 'margin-left':'1%' , 'margin-right':'3.5%','display': 'inline-block','height':'125px','vertical-align': 'top','border-style': 'solid','border-color':'#f2f2f2','padding':'0.5%','border-width': 'thin'})

    ],style={'width':'100%', 'textAlign': 'center'}),
                            
    #Network Graph
    html.Div([
        dcc.Graph(id='Graph2')
    ], style={'width':'45.5%', 'margin-top':'1%','margin-left':'2%','textAlign': 'center', 'display': 'inline-block','border-style': 'solid','border-color':'#f2f2f2','padding':'0.5%','border-width': 'thin'}),

    #TSNE Graph
    html.Div([
        dcc.Graph(id='graph', animate=True)#]), className="row")
    ], style={'width':'45.5%', 'margin-top':'1%','margin-left':'1%','margin-right':'2%','textAlign': 'center', 'display': 'inline-block','border-style': 'solid','border-color':'#f2f2f2','padding':'0.5%','border-width': 'thin'}),
    
    
    #Heatmaps
    #Heatmap1 X0 
    html.Div([
        html.Div([
            dcc.Graph(
                id='heatmap_x0'
            ),
            dcc.Graph(
                id='heatmap_x0_sel'
            )
        ], style={'width': '36%', 'display': 'inline-block'}),
        
        #Heatmap1 X1
        html.Div([
            dcc.Graph(
                id='heatmap_x1'
            ),
            dcc.Graph(
                id='heatmap_x1_sel'
            )
        ],style={'width': '34%', 'display': 'inline-block'}),
        
        #Heatmap1 X1
        html.Div([
            dcc.Graph(
                id='heatmap_x2'
            ),
            dcc.Graph(
                id='heatmap_x2_sel',
                figure={
                    'data': [{
                        'type': 'heatmap',
                        'colorscale':'Reds',
                        'showscale': False
                    }]
                }
            )
        ], style={'width': '30%', 'display': 'inline-block'})

    ], style={'width':'93.5%', 'textAlign': 'center','border-style': 'solid',
    'margin-left':'2%','margin-right':'2%','margin-top':'1%',
    'border-color':'#f2f2f2','padding':'0.5%','border-width': 'thin'})
], style={'width':'100%', 'Align': 'center','font-family':'Arial'})


@app.callback(
    Output('num_nodes', 'children'),
    [Input('dropdown-dataset', 'value')])
def update_numnodes(dataset):
    global dataset_dict
    graph_fname = dataset_dict[dataset]['graph']
    
    # get graph information
    df_graph = pd.read_csv(graph_fname, header=None)
    df_subset = df_graph[df_graph[2] == 0]

    num_nodes = len(df_subset[3].unique())

    return [
        json.dumps(str(num_nodes), indent=2)
    ]

@app.callback(
    Output('num_edges', 'children'),
    [Input('dropdown-dataset', 'value')])
def update_numnodes(dataset):
    global dataset_dict
    edge_fname = dataset_dict[dataset]['edges']
    
    # get graph information
    df_edges = pd.read_csv(edge_fname, header=None)
    
    return [
        json.dumps(str(df_edges.shape[0]), indent=2)
    ]

@app.callback(
    Output('num_features', 'children'),
    [Input('dropdown-dataset', 'value')])
def update_numnodes(dataset):
    global dataset_dict
    df = dataset_dict[dataset]['X0']
    
    # get graph information
    return [
        json.dumps(str(df.shape[1]), indent=2)
    ]

@app.callback(Output('auto-stepper', 'interval'),
               [Input('tabs', 'value')])
def disable_interval(tab):
    if tab == "pause":
        return 60*60*1000
    else:
        return 1 * 1000

@app.callback(Output('epochs-slider', 'value'),
             [Input('auto-stepper', 'n_intervals')])
def on_click(n_intervals):
    if n_intervals is None:
        return 0
    else:
        return ((n_intervals % 21) * 10)

@app.callback(Output('heatmap_x0', 'figure'),
             [Input('epochs-slider', 'value'),
              Input('graph', 'clickData')],
             [State('dropdown-dataset', 'value')])
def update_heatmap_x0(Epoch, clickData, dataset):
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }
    global dataset_dict
        
    if dataset:
        df = dataset_dict[dataset]['X0']
        
        x, y, values = process_X(df)
        
        data_dict = {
            'z': values,
            'x': x,
            'y': y,
            'type': 'heatmap',
            'colorscale':'Reds'
        }
    
        figure['data'].append(data_dict)

        if clickData is not None:
            line_val = int(clickData['points'][0]['text'])
            ll = line_val + 0.5
            ul = line_val + 1.5
            figure['data'].append(go.Scatter(
                    x = [len(x) + 0.5, 0.5, 0.5, len(x) + 0.5, len(x) + 0.5],
                    y = [ul, ul, ll, ll, ul],
                    line = dict(width=2, color='#000'),
                    mode = 'lines'
                ))

        # fill in most of layout
        figure['layout']['xaxis'] = {'autorange': True, 'nticks': 20,'title':"Features"}
        figure['layout']['yaxis'] = {'autorange': True, 'nticks': 25,'title':"Nodes"}
        figure['layout']['hovermode'] = 'closest'
        figure['layout']['title'] = 'Feature Map Input (X0)'
        figure['layout']['titlefont'] = dict(size=16)

        return figure

    else:

        return {}

@app.callback(Output('heatmap_x0_sel', 'figure'),
             [Input('epochs-slider', 'value'),
              Input('graph', 'clickData')],
             [State('dropdown-dataset', 'value')])
def update_heatmap_x0_sel(Epoch, clickData, dataset):
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }
    global dataset_dict
        
    if (dataset) and (clickData is not None):
        df = dataset_dict[dataset]['X0']
        line_val = int(clickData['points'][0]['text'])

        df_sel = df.loc[:,[line_val]]
        x, y, values = process_X(df_sel.T)
        
        data_dict = {
            'z': values,
            'x': x,
            'y': [line_val + 1],
            'type': 'heatmap',
            'colorscale':'Reds',
            'showscale': False
        }
    
        figure['data'].append(data_dict)

        # fill in most of layout
        figure['layout']['xaxis'] = {'autorange': True, 'nticks': 20,'title':"Features"}
        figure['layout']['yaxis'] = {'autorange': True, 'nticks': 2}
        figure['layout']['hovermode'] = 'closest'
        figure['layout']['height'] = 200
        figure['layout']['title'] = 'X0 Selection'
        figure['layout']['titlefont'] = dict(size=16)

        return figure

    else:

        return {}

@app.callback(Output('heatmap_x1', 'figure'),
            [Input('epochs-slider', 'value'),
              Input('graph', 'clickData')],
             [State('dropdown-dataset', 'value')])
def update_heatmap_x1(Epoch, clickData, dataset):
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }
    global dataset_dict
    if dataset:
        df = dataset_dict[dataset]['X1']
        
        x, y, values = process_X(df, Epoch)
        data_dict = {
            'z': values,
            'x': x,
            'y': y,
            'type': 'heatmap',
            'colorscale':'Reds'
        }
        
        figure['data'].append(data_dict)
        
        if clickData is not None:
            line_val = int(clickData['points'][0]['text'])
            ll = line_val + 0.5
            ul = line_val + 1.5
            figure['data'].append(go.Scatter(
                    x = [len(x) + 0.5, 0.5, 0.5, len(x) + 0.5, len(x) + 0.5],
                    y = [ul, ul, ll, ll, ul],
                    line = dict(width=2, color='#000'),
                    mode = 'lines'
                ))

    
        # fill in most of layout
        figure['layout']['xaxis'] = {'tickmode': 'linear', 'tick0': 17, 'dtick': 2,'title':"Hidden Layer Features"}
        figure['layout']['yaxis'] = {'autorange': True, 'tick0': 25}
        figure['layout']['hovermode'] = 'closest'
        figure['layout']['title'] = 'Hidden Layer Feature Map (X1)'
        figure['layout']['titlefont'] = dict(size=16)

        return figure

    else:

        return {}


@app.callback(Output('heatmap_x1_sel', 'figure'),
             [Input('epochs-slider', 'value'),
              Input('graph', 'clickData')],
             [State('dropdown-dataset', 'value')])
def update_heatmap_x1_sel(Epoch, clickData, dataset):
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }
    global dataset_dict
        
    if (dataset) and (clickData is not None):
        df = dataset_dict[dataset]['X1']
        line_val = int(clickData['points'][0]['text'])

        df_sel = df[df.index == line_val]
        x, y, values = process_X(df_sel, Epoch)

        data_dict = {
            'z': values,
            'x': x,
            'y': [line_val + 1],
            'type': 'heatmap',
            'colorscale':'Reds',
            'showscale': False
        }
    
        figure['data'].append(data_dict)

        # fill in most of layout
        figure['layout']['xaxis'] = {'tickmode': 'linear', 'tick0': 17, 'dtick': 2,'title':"Hidden Layer Features"}
        figure['layout']['yaxis'] = {'autorange': True, 'nticks': 2}
        figure['layout']['hovermode'] = 'closest'
        figure['layout']['height'] = 200
        figure['layout']['title'] = 'X1 Selection'
        figure['layout']['titlefont'] = dict(size=16)

        return figure

    else:

        return {}

@app.callback(Output('heatmap_x2', 'figure'),
            [Input('epochs-slider', 'value'),
              Input('graph', 'clickData')],
             [State('dropdown-dataset', 'value')])
def update_heatmap_x2(Epoch, clickData, dataset):
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }
    if dataset:
        global dataset_dict
        df = dataset_dict[dataset]['X2']
        
        x, y, values = process_X(df, Epoch)
        data_dict = {
            'z': values,
            'x': x,
            'y': y,
            'type': 'heatmap',
            'colorscale':'Reds'
        }
        
        figure['data'].append(data_dict)

        if clickData is not None:
            line_val = int(clickData['points'][0]['text'])
            ll = line_val + 0.5
            ul = line_val + 1.5
            figure['data'].append(go.Scatter(
                    x = [len(x) + 0.5, 0.5, 0.5, len(x) + 0.5, len(x) + 0.5],
                    y = [ul, ul, ll, ll, ul],
                    line = dict(width=2, color='#000'),
                    mode = 'lines'
                ))

    
        # fill in most of layout
        figure['layout']['xaxis'] = {'tickmode': 'linear', 'tick0': 8, 'dtick': 1,'title':"Output Layer Features"}
        figure['layout']['yaxis'] = {'autorange': True, 'tick0': 25,'title':"Nodes"}
        figure['layout']['hovermode'] = 'closest'
        figure['layout']['title'] = 'Output Layer Feature Map (X2)'
        figure['layout']['titlefont'] = dict(size=16)

        return figure
    
    else:
        
        return {}


@app.callback(Output('heatmap_x2_sel', 'figure'),
             [Input('epochs-slider', 'value'),
              Input('graph', 'clickData')],
             [State('dropdown-dataset', 'value')])
def update_heatmap_x2_sel(Epoch, clickData, dataset):
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }
    global dataset_dict
        
    if (dataset) and (clickData is not None):
        df = dataset_dict[dataset]['X2']
        line_val = int(clickData['points'][0]['text'])

        df_sel = df[df.index == line_val]
        x, y, values = process_X(df_sel, Epoch)
        data_dict = {
            'z': values,
            'x': x,
            'y': [line_val + 1],
            'type': 'heatmap',
            'colorscale':'Reds',
            'showscale': False
        }
    
        figure['data'].append(data_dict)

        # fill in most of layout
        figure['layout']['xaxis'] = {'tickmode': 'linear', 'tick0': 8, 'dtick': 1,'title':"Output Layer Features"}
        figure['layout']['yaxis'] = {'autorange': True, 'nticks': 1}
        figure['layout']['hovermode'] = 'closest'
        figure['layout']['height'] = 200
        figure['layout']['title'] = 'X2 Selection'
        figure['layout']['titlefont'] = dict(size=16)

        return figure

    else:

        return {}

@app.callback(Output('graph', 'figure'),
            [Input('epochs-slider', 'value'),
             Input('graph', 'clickData')],
             [State('dropdown-dataset', 'value')])
def update_figure(epoch, selectedPoint, dataset):
    # make figure
    figure = {
        'data': [],
        'layout': {},
        'frames': []
    }
    selectedPointIndex = []
    selectedTrace = 0
    if selectedPoint is not None:
        selectedTrace = [selectedPoint['points'][0]['curveNumber']]
        selectedPointIndex = [selectedPoint['points'][0]['pointIndex']]

    if dataset:
        global dataset_dict
        df = dataset_dict[dataset]['epochs']

        # make data
        groups = [int(i) for i in sorted(df["Group"].unique())]
        for group in groups:
            dataset_by_epoch = df[df['Epochs'] == epoch]
            dataset_by_epoch_and_grp = dataset_by_epoch[dataset_by_epoch['Group'] == group]

            data_dict = {
                'x': list(dataset_by_epoch_and_grp['Dim_1']),
                'y': list(dataset_by_epoch_and_grp['Dim_2']),
                'mode': 'markers',
                'opacity': 0.8,
                'text': list(dataset_by_epoch_and_grp['NodeID']),
                'marker': {
                        'size': 7,
                        'line': {'width': 0.5, 'color': 'white'},
                        'colorscale':'Viridis', #not working
                        'color': 'Group'    #not working
                        
                },
                'name': group
            }
            
            figure['data'].append(data_dict)

        figure['data'][selectedTrace].update(
            selected = dict(marker=dict(size= 20)),
            selectedpoints = selectedPointIndex
        )
        
        # fill in most of layout
        figure['layout']['xaxis'] = {'autorange': True}
        figure['layout']['yaxis'] = {'autorange': True,'type': 'linear'}
        #figure['layout']['colorscale']='Viridis',
        figure['layout']['hovermode'] = 'closest'
        figure['layout']['title'] = 'Hidden Layer Activations - Select to see heatmaps below'
        figure['layout']['titlefont'] = dict(size=16)

        return figure

#Graph Callback function
#Graph Callback function

@app.callback(Output('Graph2', 'figure'),
            [Input('epochs-slider', 'value'),
            Input('graph','clickData')],
            [State('dropdown-dataset', 'value')])
def update_network_graph(epoch, selectedPoint, dataset):

    
    #GRAPH DATA #####
    myDict={}
    
    global dataset_dict
    graph_fname = dataset_dict[dataset]['graph']
    
    # get graph information
    df_graph = pd.read_csv(graph_fname, header=None)
    df_subset = df_graph[df_graph[2] == epoch]

    # get nodes position
    labels = df_subset[4].values.tolist()
    for indx, row in df_subset.iterrows():
        # disregard separating rows
        myDict[int(row[3])]=[float(row[0]),float(row[1])]

    # get edge information
    edges_fname = dataset_dict[dataset]['edges']
    df_edges = pd.read_csv(edges_fname, header=None)
    edgeData = (df_edges - 1).values.tolist()
    
    # number of nodes
    num_nodes = len(df_subset[3].unique())
    
    #create graph G
    G = nx.Graph()
    G.add_edges_from(edgeData)
    
    # pos = nx.layout.spring_layout(G) !it is getting overwritten in next line
    pos = myDict
    
    #add a pos attribute to each node
    for node in G.nodes:
        G.nodes[node]['pos'] = list(pos[node])

    ##from the docs, create a random geometric graph for test
    pos = nx.get_node_attributes(G, 'pos')
    
    dmin = 1
    ncenter = 0
    for n in pos:
        x, y = pos[n]
        d = ((x - 0.5) ** 2) + ((y - 0.5) ** 2)
        if d < dmin:
            ncenter = n
            dmin = d

    p = nx.single_source_shortest_path_length(G, ncenter)

    # changed from here 
    fig = {
        'data': [],
        'layout': {},
        'frames': []
    }
    groups = [int(i) for i in sorted(df_subset[4].unique())]

    #Create Edges
    edge_trace ={
        'x' : [],
        'y' : [],
        'line' : dict(width=0.5, color='#888'),
        'hoverinfo' : None,
        'mode' : 'lines',
        'marker': {
        'size': 7,
        'line': {'width': 0.5, 'color': 'white'},
        },
        'name':'Edges'
    }

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
          
    #giving color to groups 
    for group in groups:
        node_trace = {
        'x' :[],
        'y' :[],
        'text':[],
        'mode': 'markers',
        'opacity': 0.8,
        'marker': {
        'size': 7,
        'line': {'width': 0.5, 'color': 'white'},
        'colorscale':'Viridis', #not working
        'color': 'labels'    #not working
        },
        'name':group
       
        }
        for node in G.nodes():
            if(labels[node]==group):
                x, y = G.node[node]['pos']
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
        fig['data'].append(node_trace)
        
    selectedTrace = 0
    selectedPointIndex = []
    if selectedPoint is not None:
        selectedTrace = selectedPoint['points'][0]['curveNumber']
        selectedPointIndex = [selectedPoint['points'][0]['pointIndex']]
        
    fig['data'][selectedTrace].update(
        selected = dict(marker=dict(size= 20)),
        unselected = dict(marker=dict(size= 7)),
        selectedpoints = selectedPointIndex
    )
    
    #add color to node points
    for node, adjacencies in enumerate(G.adjacency()):
        node_info = 'Node ID: ' + str(adjacencies[0])  + \
                    '<br>Group: ' + str(labels[node]) + \
                    '<br># of Connections: ' + str(len(adjacencies[1]))
        node_trace['text'] += tuple([node_info])
        
    
  
    
    fig['data'].append(edge_trace)
    fig['layout']['xaxis'] = {'autorange': True}
    fig['layout']['yaxis'] = {'autorange': True,'type': 'linear'}
    fig['layout']['hovermode'] = 'closest'
    fig['layout']['title'] = 'Evolution of Graph Node Embedding '#with ' + str(num_nodes) + ' Nodes'
    fig['layout']['titlefont'] = dict(size=16)

# changed till here

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
