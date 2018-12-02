import dash
from dash.dependencies import Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly
import random
import plotly.graph_objs as go
from collections import deque

X = deque(maxlen=50)
X.append(1)
Y = deque(maxlen=50)
Y.append(1)

app = dash.Dash(__name__)s

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
   
    html.Div(children='A live visualization of HCL Search String Categorization', style={
        'textAlign': 'center',
        'color':colors['text']
    }),

    dcc.Graph(
        id='live-graph', 
        animate=True,
        style = {'backgroundColor':colors['background']}
    ),

    dcc.Interval(
        id='graph-update',
         interval=1*500
    ),
    ]
)

@app.callback(Output('live-graph', 'figure'),
              events=[Event('graph-update', 'interval')])
def update_graph_scatter():
    X.append(X[-1]+1)
    Y.append(Y[-1]+Y[-1]*random.uniform(-0.1,0.1))
  
    data = plotly.graph_objs.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode= 'lines+markers'
            )

    return {
        'data': [data],
        'layout' : go.Layout(xaxis=dict(range=[min(X),max(X)]),
                                                yaxis=dict(range=[min(Y),max(Y)]),)
            }



if __name__ == '__main__':
    app.run_server(debug=True)