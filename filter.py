import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

def generate_filter_components(filter_def):
  components = []
  for filter in filter_def:
    if filter['type'] == 'dropdown':
      label = html.Div(className='filterLabel', children=[html.Div(filter['label'])])
      comp = dcc.Dropdown(
                id=filter['id'],
                options=filter['options'],
                value = filter['default'],
              )
      filter_comp = html.Div(className='filter', children=[label, comp])
      components.append(filter_comp)
    
    elif filter['type'] == 'rangeslider':
      label = html.Div(className='filterLabel', children=[html.Div(filter['label'])])
      comp = dcc.RangeSlider(
                id=filter['id'],
                min=filter['min'],
                max=filter['max'],
                step=filter['step'],
                value = filter['default'],
              )
      filter_comp = html.Div(className='filter', children=[label, comp])
      components.append(filter_comp)
  return components

def generate_filter_callback_input(filter_def):
  inputs = []
  for filter in filter_def:
    inputs.append(Input(component_id=filter['id'], component_property='value'))
  return inputs

def apply_filter_to_dataframe(filter_def, filter_values, df):
  for fd, fv in zip(filter_def, filter_values):
    print(fd['id'], fv)
  return df