import dash
from dash.dependencies import Input, Output, Event, State
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque
from descriptionClassifier import classify

X = deque(maxlen=50)
X.append(1)
Y = deque(maxlen=50)
Y.append(1)

errors = [line.strip() for line in open("errors.txt", 'r')]
times = [line.strip() for line in open("times.txt", 'r')]
iterations = [line.strip() for line in open("iterations.txt",'r')]
iterations = [int(line) for line in iterations]
errors = [float(line) for line in errors]
times = [float(line) for line in times]

app = dash.Dash(__name__)
server = app.server
colors = {
    'background': '#ffffff',
    'text': '#111111'
}

app.layout = html.Div(style = {'backgroundColor': colors['background']}, children=[
    html.H1(
        children='String Categorization Error and Runtime Analysis',
        style={
            'textAlign':'center',
            'color': colors['text']
            }
        ),
   
    html.Div(children='', style={
        'textAlign': 'center',
        'color':colors['text']
    }),

      dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': iterations, 
                'y': errors, 
                'type': 'lines'},
            ],
            'layout': go.Layout(
                xaxis={'title': 'Iterations'},
                yaxis={'title': 'Percent Error'},
                hovermode='closest'
            )
        }
    ),
    dcc.Graph(
        id='example-graph2',
        figure={
            'data': [
                {'x': iterations, 
                'y': times, 
                'type': 'lines'},
            ],
            'layout': go.Layout(
                xaxis={'title': 'Iterations'},
                yaxis={'title': 'Runtime (Seconds)'},
                hovermode='closest'
            )
        }
    ),
    dcc.Interval(
        id='graph-update',
         interval=1*500
    ),
    dcc.Input(id='input-1-state', type='text', value=''),
    html.Button(id='submit-button', n_clicks=0, children='Submit'),
    html.Div(id='output-state')
    ]
    
)

# @app.callback(Output('live-graph', 'figure'),
#               events=[Event('graph-update', 'interval')])
# def update_graph_scatter():
#     X.append(X[-1]+1)
#     Y.append(Y[-1]+Y[-1]*random.uniform(-0.1,0.1))
  
#     data = plotly.graph_objs.Scatter(
#             x=list(X),
#             y=list(Y),
#             name='Scatter',
#             mode= 'lines+markers'
#             )

#     return {
#         'data': [data],
#         'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
#                                                 yaxis=dict(range=[min(Y),max(Y)]),)
#             }
@app.callback(Output('output-state', 'children'),
              [Input('submit-button', 'n_clicks')],
              [State('input-1-state', 'value')])
def update_output(n_clicks, input1):
    if input1 == "":
        return ""
    return ' Your product description is \"{}. It is classified as {}!!!'.format(input1, classify(input1))


if __name__ == '__main__':
    app.run_server(debug=True)