import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
import pandas as pd

import pandas as pd
import plotly.express as px

adult1_columns = ['a1_age', 'a1_sex', 'a1_employed', 'a1_grade', 'a1_menthealth', 'a1_physhealth', 'a1_marital', 'a1_relation']
adult2_columns = ['a2_age', 'a2_sex', 'a2_employed', 'a2_grade', 'a2_menthealth', 'a2_physhealth', 'a2_marital', 'a2_relation']
gen_adult_info = ['family_r', 'famcount', 'totkids_r', 'k9q40', 'k9q41', 'k7q33', 'k8q35', 'hopeful', 'k8q31', 'k8q32', 'k8q34']
gen_child_info = ['sc_age_years', 'birth_yr', 'agepos4', 'sc_race_r', 'sc_sex', 'momage', 'k2q01', 'currcov', 's4q01', 'k4q20r', 'k4q22_r', 'sc_cshcn', 'sc_k2q10', 'sc_k2q11', 'sc_k2q12', 'sc_k2q16', 'sc_k2q17', 'sc_k2q18', 'sc_k2q19', 'sc_k2q20', 'sc_k2q21', 'sc_k2q22', 'sc_k2q23', 'screentime']
child_experiences = ['ace1', 'ace3', 'ace4', 'ace5', 'ace6', 'ace7', 'ace8', 'ace9', 'ace10', 'ace12', 'bullied_r', 'bully']
phys_health = ['allergies', 'allergies_curr', 'allergies_desc', 'arthritis', 'arthritis_curr', 'arthritis_desc', 'k2q40a', 'k2q40b', 'k2q40c', 'blindness', 'blood', 'blood_desc', 'k2q61a', 'k2q61b', 'confirminjury', 'cystfib', 'k2q43b', 'k2q41a', 'k2q41b', 'k2q41c', 'k2q42a', 'k2q42b', 'k2q42c', 'genetic', 'genetic_desc', 'headache', 'headache_curr', 'headache_desc', 'heart', 'heart_curr', 'heart_desc', 'overweight']
ment_health = ['k2q31a', 'k2q31b', 'k2q31c', 'k2q31d', 'addtreat', 'k2q33a', 'k2q33b', 'k2q33c', 'k2q35a', 'k2q35b', 'k2q35c', 'autismmed', 'autismtreat', 'k2q34a', 'k2q34b', 'k2q34c', 'k2q32a', 'k2q32b', 'k2q32c', 'k2q36a', 'k2q36b', 'k2q36c', 'downsyn', 'k4q23', 'k2q60a', 'k2q60b', 'k2q60c', 'k2q30a', 'k2q30b', 'k2q30c', 'k2q37a', 'k2q37b', 'k2q37c', 'k2q38a', 'k2q38b', 'k2q38c']
other_health = ['bedtime', 'calmdown', 'clearexp', 'distracted', 'hurtsad', 'newactivity', 'playwell', 'simpleinst', 'sitstill', 'temper', 'worktofin', 'k6q70_r', 'k6q71_r', 'k6q72_r', 'k6q73_r']
columns_of_interest = adult1_columns + adult2_columns + ['sc_age_years', 'agepos4', 'sc_race_r', 'sc_sex', 'sc_cshcn', 'k2q01', 'currcov', 's4q01', 'k4q20r', 'k4q22_r', 'screentime'] + child_experiences + ment_health

all_selected_vars = adult1_columns + adult2_columns + gen_adult_info + gen_child_info + child_experiences + phys_health + ment_health + other_health
all_adult_vars = adult1_columns + adult2_columns + gen_adult_info
all_child_vars = gen_child_info + child_experiences + phys_health + ment_health + other_health
all_child_health = phys_health + ment_health + other_health

labels = ["Age of Adult 1", "Sex of Adult 1", "Employment Status of Adult 1", "Education of Adult 1", "Mental Health of Adult 1", "Physical Health of Adult 1", "Marital Status of Adult 1", "Relation of Adult 1 to Child", "Age of Adult 2", "Sex of Adult 2", "Employment Status of Adult 2", "Education of Adult 2", "Mental Health of Adult 2", "Physical Health of Adult 2", "Marital Status of Adult 2", "Relation of Adult 2 to Child", "Age of Child", "Birth Order of Child", "Race of Child", "Sex of Child", "Special Health Care Needs Status", "General Health", "Health Insurance Coverage", "Doctor Visit Within Past 12 Months", "Frequency of Preventative Doctor Visits", "Mental Health Profession Treatment Within Past 12 Months", "Amount of Screentime", "Experienced Difficulty to Cover Basics", "Experienced Divorce of Parent/Gaurdian Get", "Experienced Death of Parent/Gaurdian", "Experienced Having a Parent/Guardian in Jail", "Experienced Adults Hitting One Another at Home", "Experienced Violence as a Victim or Witness", "Lived With a Mentally Ill Individual", "Lived With a Drug/Alchohol Abuser", "Treated Unfairly Because of Race", "Treated Unfairly Because of Sexual Orientation or Gender Identity", "Bullied by Others", "Bullies Others", "Diagnosed with ADD/ADHD", "Currently has ADD/ADHD", "Severity of ADD/ADHD", "Medicated for ADD/ADHD", "Recieves Behavioral Treatment for ADD/ADHD", "Diagnosed with Anxiety", "Currently has Anxiety", "Severity of Anxiety", "Diagnosed with Autism", "Currently has Autism", "Severity of Autism", "Medicated for Autism", "Receives Behavioral Treatment for Autism", "Diagnosed with Behavior Problems", "Currently has Behavior Problems", "Severity of Behavior Problems", "Diagnosed with Depression", "Currently has Depression", "Severity of Depression", "Diagnosed with a Developmental Delay", "Currently has a Developmental Delay", "Severity of Developmental Delay", "Diagnosed with Down Syndrome", "Takes Emotion/Concentration/Behavior Medication", "Diagnosed with an Intellectual Disability", "Currently has an Intellectual Disability", "Severity of Intellectual Disability", "Diagnosed with a Learning Disability", "Currently has a Learning Disability", "Severity of Learning Disability", "Diagnosed with a Speech Disorder", "Currently has a Speech Disorder", "Severity of Speech Disorder", "Diagnosed with Tourette Syndrome", "Currently has Tourette Syndrome", "Severity of Tourette Syndrome"]

df = pd.read_csv('./data/nsch_2020_topical.csv')[columns_of_interest]
df.columns = labels
data = df
#data = data.fillna(0)
corr = data.corr()
#mask = np.triu(np.ones_like(corr, dtype=bool))
#corr = corr.mask(mask)

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.H1("National Survey of Children's Health"),
    html.H2(""),
    html.H2("Correlation Visualization Tool"),
    html.H2(""),
    html.H3("Please select varibles to be compared."),
    dcc.Checklist(
        id="all-or-none",
        options=[{"label": "Select All", "value": "All"}],
        value=[],
        labelStyle={"display": "inline-block"},
    ),
    html.P(""),
    dcc.Dropdown(
        id='variables',
        options=[{'label': x, 'value': x} 
                 for x in labels],
        value=[],
        multi=True
    ),
    html.H1(""),
    html.Div(children=[
        dcc.Graph(id="graph", style={'display': 'inline-block'}),
        dcc.Graph(id="scatter", style={'display': 'inline-block'}),
    ], style={'width': '100%', 'display': 'inline-block'})
])

@app.callback(
    Output("graph", "figure"), 
    Input("variables", "value")
)
def update_figure(vars):
    df = data[vars]
    new_corr = df.corr()
    fig = px.imshow(new_corr, labels=dict(x="", y="", color="Correlation"), x=new_corr.columns, y=new_corr.columns, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig['layout']['xaxis']['side'] = 'top'
    fig.update_layout(autosize=True, height=1250, width=1250)
    return fig

@app.callback(
    Output("scatter", "figure"), 
    Input("graph", "hoverData"),
    Input("variables", "value"),
)
def update_scatter(hov_data, vars):
    df = data[vars]
    x_value = hov_data['points'][0]['x']
    y_value = hov_data['points'][0]['y']
    fig = px.scatter(df, x=x_value, y=y_value, trendline="ols", trendline_color_override="red")
    fig.update_layout(xaxis = dict(tickmode = 'linear', tick0 = 0,dtick = 1))
    fig.update_layout(yaxis = dict(tickmode = 'linear', tick0 = 0,dtick = 1))
    fig.update_layout(autosize=True, height=750, width=750)
    return fig

@app.callback(
    Output("variables", "value"),
    [Input("all-or-none", "value")],
    [State("variables", "options")],
)
def select_all_none(all_selected, options):
    all_or_none = []
    all_or_none = [option["value"] for option in options if all_selected]
    return all_or_none


if __name__ == '__main__':
    app.run_server(debug=True)