
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

from filter import generate_filter_components, generate_filter_callback_input, apply_filter_to_dataframe

import pandas as pd
df = pd.read_csv('./data/nsch_2020_topical.csv')
print(df.head())

filter_def = [
  {
    "label": "Filter1",
    "id": 'filter_1',
    "type": "dropdown",
    "options": [{'label': 'Age', 'value': 0},
                {'label': 'Race', 'value': 1},
                {'label': 'Sex', 'value': 2},],
    "default": 0,
  },
  {
    "label": "Age",
    "id": 'filter_age',
    "type": "rangeslider",
    "min": 0,
    "max": 19,
    "step": 1,
    "default": [5,15]
  },
  
]

filters = generate_filter_components(filter_def)
inputs = generate_filter_callback_input(filter_def)


### The app

app = dash.Dash(__name__)

@app.callback(
  Output('filtered_data', 'data'),
  *inputs
)
def handle_filter_update(*args):
  print('filter_update=', args)
  filtered_df = apply_filter_to_dataframe(filter_def, args, df)
  return filtered_df.to_json(date_format='iso', orient='split')

@app.callback(
  Output('chart1', 'children'),
  Input('filtered_data', 'data')
)
def handle_chart1(jsonified_df):
  print('chart1')
  dff = pd.read_json(jsonified_df, orient='split')
  print(dff.head())
  return []

@app.callback(
  Output('chart2', 'children'),
  Input('filtered_data', 'data')
)
def handle_chart2(jsonified_df):
  print('chart2')
  dff = pd.read_json(jsonified_df, orient='split')
  print(dff.head())
  return []

server = app.server

app.layout = html.Div(
  className='app',
  children=[
    dcc.Store(id='filtered_data'),
    html.Div(
      className='navbar',
      children=[
        html.Div("National Survey of Children's Health", className='navbarTitle'),
      ]
    ),
    html.Div(
      className='mainPanel',
      children=[
        html.Div(
          className='leftPanel',
          children=[
            html.H2("Filters"),
            *filters,
            html.Div(id='filter_output')
          ]
        ),
        html.Div(
          className='contentPanel',
          children=[
            
            html.H2(""),
            html.H2("Data Overview"),
            html.Div(id='chart1'),
            html.Div(id='chart2')
          ]
        ),
      ]
    )
  ]
)


if __name__ == '__main__':
  app.run_server(debug=True)