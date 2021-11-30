import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import io
import base64
from ast import literal_eval
import itertools

"""
import seaborn as sns

def find_friendly_name(s_column, lst_column_decoder):
    s_column = s_column.upper()
    if any(s_column in s for s in lst_column_decoder):
        for s_lst_value in lst_column_decoder:
            if s_column in s_lst_value:
                #print(s_lst_value + ' ' + s_lst_value + '\n', end='\n')
                a_return = str(s_lst_value).split(' - ', maxsplit=1)
                return(a_return[1])

def rename_columns(df_data):
    for s_column in df_data:
        if s_column != 'DATE':
            s_new_column_name = find_friendly_name(s_column, lst_column_decoder)
            df_data.rename(columns={s_column : s_new_column_name}, inplace=True)
    return(df_data)

def pickle_save(s_file_location, df_to_pickle):
    pd.to_pickle(df_to_pickle, s_file_location)

def pickle_open(s_file_location):
    return(pd.read_pickle(s_file_location))

##if __name__ == '__main__':

b_create_pickle = False

# WORKING DIRECTORY
s_wk_dir = './data/'

if b_create_pickle:
    nan_value = float("NaN")

    #LOAD AND PREPARE THE FILE DESCRIPTION FILE TO USE AS DECODING THE COLUMN NAMES
    with open (s_wk_dir + 'File_Description.txt',encoding='utf-8') as f:
        lst_file_description = f.readlines()
    f.close()

    s_file_description = ' '.join(lst_file_description)
    del lst_file_description
    a_file_description = s_file_description.split('\n')
    del s_file_description

    lst_column_decoder = []
    for s in a_file_description:
        if '-' in s:
            lst_column_decoder.append(s)

    #READ FILES, SET A DATE ACCORDING FILE BEING LOADED
    df_data = pd.read_stata(s_wk_dir + 'nsch_2020_topical.dta')
                            #columns=a_columns)
    df_data['YEAR'] = '2020'
    df_temp_data = pd.read_stata(s_wk_dir + 'nsch_2019_topical.dta')
                                    #columns=a_columns)
    df_temp_data['YEAR'] = '2019'
    df_data.append(df_temp_data)
    df_temp_data = pd.read_stata(s_wk_dir + 'nsch_2018_topical.dta')
                                    #columns=a_columns)
    df_temp_data['YEAR'] = '2018'
    df_data.append(df_temp_data)

    df_temp_data = pd.read_stata(s_wk_dir + 'nsch_2017_topical.dta')
                                # columns=a_columns)
    df_temp_data['YEAR'] = '2017'
    df_data.append(df_temp_data)

    #df_data = df_data.set_index('YEAR')

    # data = dowhy.datasets.linear_dataset(
    #     beta=10,
    #     num_common_causes=5,
    #     num_instruments=2,
    #     num_samples=10000,
    #     treatment_is_binary=True)


    del df_temp_data

    df_temp_data = pd.DataFrame(df_data, columns=['YEAR'])
    df_data = rename_columns(df_data)

    df_data['YEAR'] = df_temp_data
    del df_temp_data
    #MODIFY VALUE TYPE OF DATE TO A DATETIME VALUE
    #df_data['DATE'] = pd.to_datetime(df_data['DATE']).dt.date

    # #INDEX ACCORDING TO DATE
    #df_data = df_data.set_index('YEAR')


    #CALCULATE CORRELATION
    df_corr = df_data.corr().abs()
    #PREPARE TO DISPLAY RESULTS
    df_corr_unstacked = df_corr.unstack().to_frame()

    df_corr_unstacked.rename(columns={list(df_corr_unstacked)[0]:'CORRELATION'}, inplace=True)
    pd.to_numeric(df_corr_unstacked['CORRELATION'], errors='coerce').notnull().all()

    df_corr_unstacked.dropna(subset=['CORRELATION'], inplace=True)

    df_corr_unstacked = df_corr_unstacked[df_corr_unstacked.CORRELATION > .8]
    df_corr_unstacked.reset_index(inplace=True)


    #REMOVE RECORDS WHICH LEFT AND RIGHT ARE THE SAME
    df_corr_unstacked = df_corr_unstacked[df_corr_unstacked['level_1'] != df_corr_unstacked['level_0']]


    #GET LIST OF COLUMNS WHICH HAVE CORRELATION
    lst_columns = df_corr_unstacked['level_1'].tolist()
    #CREATE A SMALLER DF WITH ONLY THOSE COLUMNS TO CHECK FOR CAUSALITY
    df_reduced_main_data = df_data[df_data.columns.intersection(lst_columns)]

    pickle_save(s_wk_dir + 'pickedmaindata',df_reduced_main_data)
    pickle_save(s_wk_dir + 'pickedcorrdata', df_corr_unstacked)
    pickle_save(s_wk_dir + 'pickedrawmaindata', df_data)


    del a_file_description, df_corr_unstacked, df_corr, df_data, df_reduced_main_data, f, s

df_reduced_main_data = pickle_open(s_wk_dir + 'pickedmaindata')
df_corr_unstacked= pickle_open(s_wk_dir + 'pickedcorrdata')
df_raw_main_data = pickle_open(s_wk_dir + 'pickedrawmaindata')

# DESCRIBE DATA

######################
#HISTOGRAM
fig_1 = px.histogram(df_raw_main_data, y='Age of Selected Child - In Years (S1)', x='YEAR')
fig_1.update_layout(bargap=0.1)
#fig_1.show()

fig_2 = px.histogram(df_raw_main_data, x='Birth Year (T1 T2 T3)')
fig_2.update_layout(bargap=0.1)
#fig_2.show()
"""

adult1_columns = ['a1_age', 'a1_sex', 'a1_employed', 'a1_grade', 'a1_menthealth', 'a1_physhealth', 'a1_marital', 'a1_relation']
adult2_columns = ['a2_age', 'a2_sex', 'a2_employed', 'a2_grade', 'a2_menthealth', 'a2_physhealth', 'a2_marital', 'a2_relation']
gen_adult_info = ['family_r', 'famcount', 'totkids_r', 'k9q40', 'k9q41', 'k7q33', 'k8q35', 'hopeful', 'k8q31', 'k8q32', 'k8q34']
gen_child_info = ['sc_age_years', 'birth_yr', 'agepos4', 'sc_race_r', 'sc_sex', 'momage', 'k2q01', 'currcov', 's4q01', 'k4q20r', 'k4q22_r', 'sc_cshcn', 'sc_k2q10', 'sc_k2q11', 'sc_k2q12', 'sc_k2q16', 'sc_k2q17', 'sc_k2q18', 'sc_k2q19', 'sc_k2q20', 'sc_k2q21', 'sc_k2q22', 'sc_k2q23', 'screentime']
child_experiences = ['ace1', 'ace3', 'ace4', 'ace5', 'ace6', 'ace7', 'ace8', 'ace9', 'ace10', 'ace12', 'bullied_r', 'bully']
phys_health = ['allergies', 'allergies_curr', 'allergies_desc', 'arthritis', 'arthritis_curr', 'arthritis_desc', 'k2q40a', 'k2q40b', 'k2q40c', 'blindness', 'blood', 'blood_desc', 'k2q61a', 'k2q61b', 'confirminjury', 'cystfib', 'k2q43b', 'k2q41a', 'k2q41b', 'k2q41c', 'k2q42a', 'k2q42b', 'k2q42c', 'genetic', 'genetic_desc', 'headache', 'headache_curr', 'headache_desc', 'heart', 'heart_curr', 'heart_desc', 'overweight']
ment_health = ['k2q31a', 'k2q31b', 'k2q31c', 'k2q31d', 'addtreat', 'k2q33a', 'k2q33b', 'k2q33c', 'k2q35a', 'k2q35b', 'k2q35c', 'autismmed', 'autismtreat', 'k2q34a', 'k2q34b', 'k2q34c', 'k2q32a', 'k2q32b', 'k2q32c', 'k2q36a', 'k2q36b', 'k2q36c', 'downsyn', 'k4q23', 'k2q60a', 'k2q60b', 'k2q60c', 'k2q30a', 'k2q30b', 'k2q30c', 'k2q37a', 'k2q37b', 'k2q37c', 'k2q38a', 'k2q38b', 'k2q38c']
other_health = ['bedtime', 'calmdown', 'clearexp', 'distracted', 'hurtsad', 'newactivity', 'playwell', 'simpleinst', 'sitstill', 'temper', 'worktofin', 'k6q70_r', 'k6q71_r', 'k6q72_r', 'k6q73_r']

all_selected_vars = adult1_columns + adult2_columns + gen_adult_info + gen_child_info + child_experiences + phys_health + ment_health + other_health
all_adult_vars = adult1_columns + adult2_columns + gen_adult_info
all_child_vars = gen_child_info + child_experiences + phys_health + ment_health + other_health
all_child_health = phys_health + ment_health + other_health

columns_of_interest = adult1_columns + adult2_columns + ['sc_age_years', 'agepos4', 'sc_race_r', 'sc_sex', 'sc_cshcn', 'k2q01', 'currcov', 's4q01', 'k4q20r', 'k4q22_r', 'screentime'] + child_experiences + ment_health
labels = ["Age of Adult 1", "Sex of Adult 1", "Employment Status of Adult 1", "Education of Adult 1", "Mental Health of Adult 1", "Physical Health of Adult 1", "Marital Status of Adult 1", "Relation of Adult 1 to Child", "Age of Adult 2", "Sex of Adult 2", "Employment Status of Adult 2", "Education of Adult 2", "Mental Health of Adult 2", "Physical Health of Adult 2", "Marital Status of Adult 2", "Relation of Adult 2 to Child", "Age of Child", "Birth Order of Child", "Race of Child", "Sex of Child", "Special Health Care Needs Status", "General Health", "Health Insurance Coverage", "Doctor Visit Within Past 12 Months", "Frequency of Preventative Doctor Visits", "Mental Health Profession Treatment Within Past 12 Months", "Amount of Screentime", "Experienced Difficulty to Cover Basics", "Experienced Divorce of Parent/Gaurdian Get", "Experienced Death of Parent/Gaurdian", "Experienced Having a Parent/Guardian in Jail", "Experienced Adults Hitting One Another at Home", "Experienced Violence as a Victim or Witness", "Lived With a Mentally Ill Individual", "Lived With a Drug/Alchohol Abuser", "Treated Unfairly Because of Race", "Treated Unfairly Because of Sexual Orientation or Gender Identity", "Bullied by Others", "Bullies Others", "Diagnosed with ADD/ADHD", "Currently has ADD/ADHD", "Severity of ADD/ADHD", "Medicated for ADD/ADHD", "Recieves Behavioral Treatment for ADD/ADHD", "Diagnosed with Anxiety", "Currently has Anxiety", "Severity of Anxiety", "Diagnosed with Autism", "Currently has Autism", "Severity of Autism", "Medicated for Autism", "Receives Behavioral Treatment for Autism", "Diagnosed with Behavior Problems", "Currently has Behavior Problems", "Severity of Behavior Problems", "Diagnosed with Depression", "Currently has Depression", "Severity of Depression", "Diagnosed with a Developmental Delay", "Currently has a Developmental Delay", "Severity of Developmental Delay", "Diagnosed with Down Syndrome", "Takes Emotion/Concentration/Behavior Medication", "Diagnosed with an Intellectual Disability", "Currently has an Intellectual Disability", "Severity of Intellectual Disability", "Diagnosed with a Learning Disability", "Currently has a Learning Disability", "Severity of Learning Disability", "Diagnosed with a Speech Disorder", "Currently has a Speech Disorder", "Severity of Speech Disorder", "Diagnosed with Tourette Syndrome", "Currently has Tourette Syndrome", "Severity of Tourette Syndrome"]

columns_of_interest2 = adult1_columns + adult2_columns + ['sc_age_years', 'agepos4', 'sc_race_r', 'sc_sex', 'sc_cshcn', 'k2q01', 'currcov', 's4q01', 'k4q20r', 'k4q22_r', 'screentime'] + child_experiences + phys_health +ment_health
labels2 = ["Age of Adult 1", "Sex of Adult 1", "Employment Status of Adult 1", "Education of Adult 1", "Mental Health of Adult 1", "Physical Health of Adult 1", "Marital Status of Adult 1", "Relation of Adult 1 to Child", "Age of Adult 2", "Sex of Adult 2", "Employment Status of Adult 2", "Education of Adult 2", "Mental Health of Adult 2", "Physical Health of Adult 2", "Marital Status of Adult 2", "Relation of Adult 2 to Child", "Age of Child", "Birth Order of Child", "Race of Child", "Sex of Child", "Special Health Care Needs Status", "General Health", "Health Insurance Coverage", "Doctor Visit Within Past 12 Months", "Frequency of Preventative Doctor Visits", "Mental Health Profession Treatment Within Past 12 Months", "Amount of Screentime", "Experienced Difficulty to Cover Basics", "Experienced Divorce of Parent/Gaurdian Get", "Experienced Death of Parent/Gaurdian", "Experienced Having a Parent/Guardian in Jail", "Experienced Adults Hitting One Another at Home", "Experienced Violence as a Victim or Witness", "Lived With a Mentally Ill Individual", "Lived With a Drug/Alchohol Abuser", "Treated Unfairly Because of Race", "Treated Unfairly Because of Sexual Orientation or Gender Identity", "Bullied by Others", "Bullies Others", "Diagnosed with Allergies", "Currently has Allergies", "Severity of Allergies", "Diagnosed with Arthritis", "Currently has Arthritis", "Severity of Arthritis", "Diagnosed with Asthma", "Currently has Asthma", "Severity of Asthma", "Blindness", "Diagnosed with a Blood Disorder", "Severity of Blood Disorder", "Diagnosed with Cerebral Palsy", "Currently has Cerebral Palsy", "Confirmed Concussion/Brain Injury", "Diagnosed with Cystic Fibrosis", "Deafness", "Diagnosed with Diabetes", "Currently has Diabetes", "Severity of Diabetes", "Diagnosed with Epilepsy", "Currently has Epilepsy", "Severity of Epilepsy", "Diagnosed with a Genetic Condition", "Severity of Genetic Condition", "Diagnosed with Headaches", "Currently has Headaches", "Severity of Headaches", "Diagnosed with a Heart Condition", "Currently has a Heart Condition", "Severity of Heart Condition", "Overweight", "Diagnosed with ADD/ADHD", "Currently has ADD/ADHD", "Severity of ADD/ADHD", "Medicated for ADD/ADHD", "Recieves Behavioral Treatment for ADD/ADHD", "Diagnosed with Anxiety", "Currently has Anxiety", "Severity of Anxiety", "Diagnosed with Autism", "Currently has Autism", "Severity of Autism", "Medicated for Autism", "Receives Behavioral Treatment for Autism", "Diagnosed with Behavior Problems", "Currently has Behavior Problems", "Severity of Behavior Problems", "Diagnosed with Depression", "Currently has Depression", "Severity of Depression", "Diagnosed with a Developmental Delay", "Currently has a Developmental Delay", "Severity of Developmental Delay", "Diagnosed with Down Syndrome", "Takes Emotion/Concentration/Behavior Medication", "Diagnosed with an Intellectual Disability", "Currently has an Intellectual Disability", "Severity of Intellectual Disability", "Diagnosed with a Learning Disability", "Currently has a Learning Disability", "Severity of Learning Disability", "Diagnosed with a Speech Disorder", "Currently has a Speech Disorder", "Severity of Speech Disorder", "Diagnosed with Tourette Syndrome", "Currently has Tourette Syndrome", "Severity of Tourette Syndrome"]

"""
df = pd.read_csv('./data/nsch_2020_topical.csv')[columns_of_interest]
df.columns = labels
data = df
#data = data.fillna(0)

corr = data.corr()
#mask = np.triu(np.ones_like(corr, dtype=bool))
#corr = corr.mask(mask)
"""

df2 = pd.read_csv('./data/nsch_2020_topical.csv')[columns_of_interest2]
df2.columns = labels2
data2 = df2
#data2 = data2.fillna(0)

corr2 = data2.corr()
#mask2 = np.triu(np.ones_like(corr2, dtype=bool))
#corr2 = corr2.mask(mask2)

"""
fig_1_1 = px.histogram(data2, x="Age of Child")
fig_1_1.update_layout(bargap=0.1)
fig_1_1.update_layout(autosize=True, height=500, width=1000)

sex_labels = ["Male", "Female"]
sex_count = data2["Sex of Child"].value_counts().to_list()
fig_2_2 = go.Figure(data=[go.Pie(labels=sex_labels, values=sex_count, textinfo='label+percent', insidetextorientation='radial')])
fig_2_2.update_layout(autosize=True, height=500, width=750)

race_labels = ["White", "Black or African American", "American Indian or Alaska Native", "Asian", "Native Hawaiian or Other Pacific Islander", "Two or More Races"]
race_count = data2["Race of Child"].value_counts().to_list()
fig_3_3 = go.Figure(data=[go.Pie(labels=race_labels, values=race_count, textinfo='label+percent', insidetextorientation='horizontal')])
fig_3_3.update_layout(autosize=True, height=500, width=1000)
"""

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div([
    html.H1("National Survey of Children's Health"),
    html.H2(""),
    html.H2("Data Overview"),
    html.H3("Select whether to view data by Child Age, Race, or Sex"),
    dcc.Dropdown(
        id='data_overview',
        options=[
            {'label': 'Age', 'value': 0},
            {'label': 'Race', 'value': 1},
            {'label': 'Sex', 'value': 2},
        ],
        value = 0
    ),
    dcc.Graph(id="overview"),
    html.H1(""),
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
        options=[{'label': x, 'value': x} for x in labels2],
        value=["Severity of Allergies", "Severity of Arthritis", "Severity of Asthma", "Severity of Blood Disorder", "Severity of Diabetes", "Severity of Epilepsy", "Severity of Genetic Condition", "Severity of Headaches", "Severity of Heart Condition", "Severity of ADD/ADHD", "Severity of Anxiety", "Severity of Autism", "Severity of Behavior Problems", "Severity of Depression", "Severity of Developmental Delay", "Severity of Intellectual Disability", "Severity of Learning Disability", "Severity of Speech Disorder", "Severity of Tourette Syndrome"],
        multi=True
    ),
    html.H1(""),
    html.H3("Please select any desired filters."),
    html.H1(""),
    html.H4("Age:"),
    dcc.Dropdown(
        id='age_filter',
        options=[
            {'label': 'All', 'value': -1},
            {'label': '0', 'value': 0},
            {'label': '1', 'value': 1},
            {'label': '2', 'value': 2},
            {'label': '3', 'value': 3},
            {'label': '4', 'value': 4},
            {'label': '5', 'value': 5},
            {'label': '6', 'value': 6},
            {'label': '7', 'value': 7},
            {'label': '8', 'value': 8},
            {'label': '9', 'value': 9},
            {'label': '10', 'value': 10},
            {'label': '11', 'value': 11},
            {'label': '12', 'value': 12},
            {'label': '13', 'value': 13},
            {'label': '14', 'value': 14},
            {'label': '15', 'value': 15},
            {'label': '16', 'value': 16},
            {'label': '17', 'value': 17},
            ],
        value = -1
    ),
    html.H1(""),
    html.H4("Race:"),
    dcc.Dropdown(
        id='race_filter',
        options=[
            {'label': 'All', 'value': -1},
            {'label': 'White', 'value': 1},
            {'label': 'Black or African American', 'value': 2},
            {'label': 'American Indian or Alaska Native', 'value': 3},
            {'label': 'Asian', 'value': 4},
            {'label': 'Native Hawaiian or Other Pacific Islander', 'value': 5},
            {'label': 'Two or More Races', 'value': 7},
        ],
        value = -1
    ),
    html.H1(""),
    html.H4("Sex:"),
    dcc.Dropdown(
        id='sex_filter',
        options=[
            {'label': 'All', 'value': -1},
            {'label': 'Male', 'value': 1},
            {'label': 'Female', 'value': 2},
        ],
        value = -1
    ),
    html.H1(""),
    html.H3("Once generated, click on any square in the heatmap to see the variable pair's scatterplot."),
    html.H1(""),
    html.Div(children=[
        dcc.Graph(id="graph", style={'display': 'inline-block'}),
        dcc.Graph(id="scatter", style={'display': 'inline-block'}),
    ], style={'width': '100%', 'display': 'inline-block'}),
    html.H1(""),
    html.H2("Node Link Visualization Tool"),
    html.H2(""),
    html.H3("Please select varibles to include in the diagram."),
    dcc.Checklist(
        id="all-or-none2",
        options=[{"label": "Select All", "value": "All"}],
        value=[],
        labelStyle={"display": "inline-block"},
    ),
    html.P(""),
    dcc.Dropdown(
        id='variables2',
        options=[{'label': x, 'value': x} for x in labels2],
        value=["Diagnosed with ADD/ADHD", "Currently has ADD/ADHD", "Severity of ADD/ADHD", "Medicated for ADD/ADHD", "Recieves Behavioral Treatment for ADD/ADHD", "Diagnosed with Anxiety", "Currently has Anxiety", "Severity of Anxiety", "Diagnosed with Autism", "Currently has Autism", "Severity of Autism", "Medicated for Autism", "Receives Behavioral Treatment for Autism", "Diagnosed with Behavior Problems", "Currently has Behavior Problems", "Severity of Behavior Problems", "Diagnosed with Depression", "Currently has Depression", "Severity of Depression", "Diagnosed with a Developmental Delay", "Currently has a Developmental Delay", "Severity of Developmental Delay", "Diagnosed with Down Syndrome", "Takes Emotion/Concentration/Behavior Medication", "Diagnosed with an Intellectual Disability", "Currently has an Intellectual Disability", "Severity of Intellectual Disability", "Diagnosed with a Learning Disability", "Currently has a Learning Disability", "Severity of Learning Disability", "Diagnosed with a Speech Disorder", "Currently has a Speech Disorder", "Severity of Speech Disorder", "Diagnosed with Tourette Syndrome", "Currently has Tourette Syndrome", "Severity of Tourette Syndrome"],
        multi=True
    ),
    html.P(""),
    html.H3("Please select correlation strength and whether correlations greater than or less than the correlation strength are displayed."),
    html.H1(""),
    dcc.RadioItems(
        id='greater_less',
        options=[
            {'label': 'Greater Than', 'value': 0},
            {'label': 'Less Than', 'value': 1}
        ],
        value = 0,
        labelStyle={'display': 'flex'}
    ),
    html.H1(""),
    dcc.Slider(
        id="corr_strength",
        min=-1,
        max=1,
        step=.1,
        marks={
        -1: '-1',
        -0.5: '-0.5',
        0: '0',
        0.5: '0.5',
        1: '1'
        },
        value=.5,
        tooltip={"placement": "bottom", "always_visible": True},
    ),
    html.H1(""),
    html.Img(id="node_link"),
    html.H1(""),
    html.H2("Multiple Regression Visualization Tool"),
    html.H2(""),
    html.H3("Please select three varibles to include in the multiple regression analysis."),
    dcc.Dropdown(
        id='x_axis',
        options=[{'label': x, 'value': x} for x in labels2],
        value = "Medicated for Autism"
    ),
    dcc.Dropdown(
        id='y_axis',
        options=[{'label': x, 'value': x} for x in labels2],
        value = 'Receives Behavioral Treatment for Autism'
    ),
    dcc.Dropdown(
        id='z_axis',
        options=[{'label': x, 'value': x} for x in labels2],
        value = 'Severity of Autism'
    ),
    html.H1(""),
    dcc.Graph(id="multi_reg"),
    html.H1(""),
])

""" ### Removed for taking too long
    html.H2("Multiple Regression Visualization Tool 2"),
    html.H2(""),
    html.H3("Please select a list of at least three vaiables to include in the multiple regression analysis."),
    html.H4("The combination with the highest R^2 value will then be displayed below."),
    dcc.Checklist(
        id="all-or-none3",
        options=[{"label": "Select All", "value": "All"}],
        value=[],
        labelStyle={"display": "inline-block"},
    ),
    html.P(""),
    dcc.Dropdown(
        id='variables3',
        options=[{'label': x, 'value': x} for x in labels2],
        value=["Diagnosed with Autism", "Currently has Autism", "Severity of Autism", "Medicated for Autism", "Receives Behavioral Treatment for Autism", "Diagnosed with a Developmental Delay", "Currently has a Developmental Delay", "Severity of Developmental Delay", "Diagnosed with an Intellectual Disability", "Currently has an Intellectual Disability", "Severity of Intellectual Disability", "Diagnosed with a Learning Disability", "Currently has a Learning Disability", "Severity of Learning Disability", "Diagnosed with a Speech Disorder", "Currently has a Speech Disorder", "Severity of Speech Disorder"],
        multi=True
    ),
    dcc.Graph(id="multi_reg2"),
    html.H1(""),
"""

"""
    dcc.Graph(figure = fig_1_1),
    html.H1(""),
    html.H3("Data by Child Sex & Race"),
    html.Div(children=[
        dcc.Graph(figure = fig_2_2, style={'display': 'inline-block'}),
        dcc.Graph(figure = fig_3_3, style={'display': 'inline-block'}),
    ], style={'width': '100%', 'display': 'inline-block'}),
"""

### Data overview
@app.callback(
    Output("overview", "figure"), 
    Input("data_overview", "value"),
)
def update_overview(var):
    if var == 0:
        fig = px.histogram(data2, x="Age of Child")
        fig.update_layout(bargap=0.1)
        fig.update_layout(autosize=True, height=500, width=1000)
        return fig
    elif var == 1:
        race_labels = ["White", "Black or African American", "American Indian or Alaska Native", "Asian", "Native Hawaiian or Other Pacific Islander", "Two or More Races"]
        race_count = data2["Race of Child"].value_counts().to_list()
        fig = go.Figure(data=[go.Pie(labels=race_labels, values=race_count, textinfo='label+percent', insidetextorientation='horizontal')])
        fig.update_layout(autosize=True, height=500, width=1000)
        return fig
    elif var == 2:
        sex_labels = ["Male", "Female"]
        sex_count = data2["Sex of Child"].value_counts().to_list()
        fig = go.Figure(data=[go.Pie(labels=sex_labels, values=sex_count, textinfo='label+percent', insidetextorientation='radial')])
        fig.update_layout(autosize=True, height=500, width=750)
        return fig

### Makes heatmap based on selected variables

@app.callback(
    Output("graph", "figure"), 
    Input("variables", "value"),
    Input("age_filter", "value"),
    Input("race_filter", "value"), 
    Input("sex_filter", "value"),
)
def update_figure(vars, age, race, sex):
    if age == -1 and race == -1 and sex == -1:
        df = data2[vars]
    elif age == -1 and race == -1 and sex != -1:
        df = data2[data2["Sex of Child"] == sex]
        df = df[vars]
    elif age == -1 and sex == -1 and race != -1:
        df = data2[data2["Race of Child"] == race]
        df = df[vars]
    elif race == -1 and sex == -1 and age != -1:        
        df = data2[data2["Age of Child"] == age]
        df = df[vars]
    elif age == -1 and race != -1 and sex != -1:
        df = data2[(data2["Race of Child"] == race) & (data2["Sex of Child"] == sex)]
        df = df[vars]
    elif race == -1 and age != -1 and sex != -1:
        df = data2[(data2["Age of Child"] == age) & (data2["Sex of Child"] == sex)]
        df = df[vars]
    elif sex == -1 and age != -1 and race != -1:
        df = data2[(data2["Age of Child"] == age) & (data2["Race of Child"] == race)]
        df = df[vars]
    elif age != -1 and race != -1 and sex != -1:
        df = data2[(data2["Age of Child"] == age) & (data2["Race of Child"] == race) & (data2["Sex of Child"] == sex)]
        df = df[vars]
    new_corr = df.corr()
    fig = px.imshow(new_corr, labels=dict(x="", y="", color="Correlation"), x=new_corr.columns, y=new_corr.columns, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig['layout']['xaxis']['side'] = 'top'
    fig.update_layout(autosize=True, height=1250, width=1250)
    return fig    

### Callback that allows you to click on heatmap to get scatterplot of variable pair

@app.callback(
    Output("scatter", "figure"), 
    Input("graph", "clickData"),
    Input("variables", "value"),
    Input("age_filter", "value"),
    Input("race_filter", "value"), 
    Input("sex_filter", "value"),
)
def update_scatter(click_data, vars, age, race, sex):
    if age == -1 and race == -1 and sex == -1:
        df = data2[vars]
    elif age == -1 and race == -1 and sex != -1:
        df = data2[data2["Sex of Child"] == sex]
        df = df[vars]
    elif age == -1 and sex == -1 and race != -1:
        df = data2[data2["Race of Child"] == race]
        df = df[vars]
    elif race == -1 and sex == -1 and age != -1:        
        df = data2[data2["Age of Child"] == age]
        df = df[vars]
    elif age == -1 and race != -1 and sex != -1:
        df = data2[(data2["Race of Child"] == race) & (data2["Sex of Child"] == sex)]
        df = df[vars]
    elif race == -1 and age != -1 and sex != -1:
        df = data2[(data2["Age of Child"] == age) & (data2["Sex of Child"] == sex)]
        df = df[vars]
    elif sex == -1 and age != -1 and race != -1:
        df = data2[(data2["Age of Child"] == age) & (data2["Race of Child"] == race)]
        df = df[vars]
    elif age != -1 and race != -1 and sex != -1:
        df = data2[(data2["Age of Child"] == age) & (data2["Race of Child"] == race) & (data2["Sex of Child"] == sex)]
        df = df[vars]
    x_value = str(click_data['points'][0]['x'])
    y_value = str(click_data['points'][0]['y'])
    if x_value == y_value:
        fig = px.scatter(df, x=x_value, y=y_value, trendline="ols", trendline_color_override="red")
        fig.update_layout(xaxis = dict(tickmode = 'linear', tick0 = 0,dtick = 1))
        fig.update_layout(yaxis = dict(tickmode = 'linear', tick0 = 0,dtick = 1))
        fig.update_layout(autosize=True, height=1000, width=1000)
        return fig
    else:
        df = df[[x_value, y_value]].copy()
        df['combo'] = df[x_value].astype(str) + df[y_value].astype(str)
        df['frequency'] = df['combo'].map(df['combo'].value_counts())
        freq = df['frequency'].tolist()
        fig = px.scatter(df, x=x_value, y=y_value, trendline="ols", trendline_color_override="red", size=freq)
        fig.update_layout(xaxis = dict(tickmode = 'linear', tick0 = 0,dtick = 1))
        fig.update_layout(yaxis = dict(tickmode = 'linear', tick0 = 0,dtick = 1))
        fig.update_layout(autosize=True, height=1000, width=1000)
        return fig

### Creates node link diagrams based on selected variables

@app.callback(
    Output("node_link", "src"), 
    Input("variables2", "value"),
    Input("corr_strength", "value"),
    Input('greater_less', 'value'),
)
def update_node_link(vars, corr_strength, greater_less):
    df = data2[vars]
    new_corr = df.corr()
    
    buf = io.BytesIO() 

    links = new_corr.stack().reset_index()
    links.columns = ['var1', 'var2', 'value']
    
    if greater_less == 0:
        links_filtered=links.loc[ (links['value'] >= corr_strength) & (links['var1'] != links['var2']) ]
    elif greater_less == 1:
        links_filtered=links.loc[ (links['value'] <= corr_strength) & (links['var1'] != links['var2']) ]

    G = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2', edge_attr=True)

    fig = plt.figure("Node Link Diagram with Selected Variables",figsize=(12,8)) 
    pos = nx.spring_layout(G, k=0.20, iterations=20)
    nx.draw(G, pos, with_labels=True, node_color='orange', node_size=600, edge_color='black', linewidths=1, font_size=10)
    
    plt.savefig(buf, format = "png") 
    plt.close()
    img_data = base64.b64encode(buf.getbuffer()).decode("utf8") 
    return "data:image/png;base64,{}".format(img_data)

### Plots multiple regression based on three inputs

@app.callback(
    Output("multi_reg", "figure"),
    Input("x_axis", "value"),
    Input("y_axis", "value"),
    Input("z_axis", "value")
)
def update_multi_reg(x_var, y_var, z_var):
    mesh_size = .02
    margin = 0

    df = data2
    df = df.fillna(0)

    X = df[[x_var, y_var]]
    y = df[[z_var]]

    model = SVR(C=1.)
    model.fit(X, y)

    x_min, x_max = X.iloc[:, 0].min() - margin, X.iloc[:, 0].max() + margin
    y_min, y_max = X.iloc[:, 1].min() - margin, X.iloc[:, 1].max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)

    df2 = df[[x_var, y_var, z_var]].copy()
    df2['combo'] = df2[x_var].astype(str) + df2[y_var].astype(str) + + df2[z_var].astype(str)
    df2['frequency'] = df2['combo'].map(df2['combo'].value_counts())
    freq = df2['frequency'].tolist()

    fig = px.scatter_3d(df, x=str(X.columns[0]), y=str(X.columns[1]), z=str(y.columns[0]), hover_data={x_var: True, y_var: True, z_var: True, "Frequency": freq})
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
    fig.update_layout(autosize=True, height=1000, width=1250)
    return fig

""" ### Works but takes too long
### Finding and plotting the best multiple regression out of list of variables

@app.callback(
    Output("multi_reg2", "figure"),
    Input("variables3", "value"),
)
def update_multi_reg2(vars):
    df = data2[vars]
    df.fillna(0)
    
    vars1 = vars
    vars2 = vars
    vars3 = vars

    var_lists = [vars1, vars2, vars3]
    combos = []
    for element in itertools.product(*var_lists):
        combos.append(element)

    combos2 = []
    for combo in combos:
        if combo[0] != combo[1] and combo[0] != combo[2] and combo[1] != combo[2]:
            combos2.append(combo)

    res_lst = []
    for combo in combos2:
        y1 = combo[0]
        Y = df[[y1]]
        Y = Y.fillna(0)

        x1 = combo[1]
        x2 = combo[2]
        X = df[[x1, x2]]
        X = X.fillna(0)

        X = sm.add_constant(X)
        model = sm.OLS(Y, X).fit()
        corr = model.rsquared
        
        result = ("Combo = ", str(combo), "R-Squared = ", corr)
        res_lst.append(result)
    
    top_res = sorted(res_lst, key=lambda x: x[3], reverse=True)[:1]
    top_combo = top_res[0][1]
    top_combo = literal_eval(top_combo)

    y1 = top_combo[0]
        
    x1 = top_combo[1]
    x2 = top_combo[2]

    mesh_size = .02
    margin = 0

    X = df[[x1, x2]]
    X = X.fillna(0)
    y = df[[y1]]
    y = y.fillna(0)

    model = SVR(C=1.)
    model.fit(X, y)

    x_min, x_max = X.iloc[:, 0].min() - margin, X.iloc[:, 0].max() + margin
    y_min, y_max = X.iloc[:, 1].min() - margin, X.iloc[:, 1].max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)

    df2 = df[[x1, x2, y1]].copy()
    df2['combo'] = df2[x1].astype(str) + df2[x2].astype(str) + + df2[y1].astype(str)
    df2['frequency'] = df2['combo'].map(df2['combo'].value_counts())
    freq = df2['frequency'].tolist()

    fig = px.scatter_3d(df, x=str(X.columns[0]), y=str(X.columns[1]), z=str(y.columns[0]), hover_data={x1: True, x2: True, y1: True, "Frequency": freq})
    fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
    fig.update_layout(autosize=True, height=1000, width=1250)
    return fig
"""

### Below are the app callbacks for the select all functionality

@app.callback(
    Output("variables", "value"),
    [Input("all-or-none", "value")],
    [State("variables", "options")],
)
def select_vars(all_selected, options):
    selected_vars = []
    if all_selected:
        selected_vars = [option["value"] for option in options]
    else:
        selected_vars = ["Severity of Allergies", "Severity of Arthritis", "Severity of Asthma", "Severity of Blood Disorder", "Severity of Diabetes", "Severity of Epilepsy", "Severity of Genetic Condition", "Severity of Headaches", "Severity of Heart Condition", "Severity of ADD/ADHD", "Severity of Anxiety", "Severity of Autism", "Severity of Behavior Problems", "Severity of Depression", "Severity of Developmental Delay", "Severity of Intellectual Disability", "Severity of Learning Disability", "Severity of Speech Disorder", "Severity of Tourette Syndrome"]
    return selected_vars
    
@app.callback(
    Output("variables2", "value"),
    [Input("all-or-none2", "value")],
    [State("variables2", "options")],
)
def select_vars2(all_selected, options):
    selected_vars = []
    if all_selected:
        selected_vars = [option["value"] for option in options]
    else:
        selected_vars = ["Diagnosed with ADD/ADHD", "Currently has ADD/ADHD", "Severity of ADD/ADHD", "Medicated for ADD/ADHD", "Recieves Behavioral Treatment for ADD/ADHD", "Diagnosed with Anxiety", "Currently has Anxiety", "Severity of Anxiety", "Diagnosed with Autism", "Currently has Autism", "Severity of Autism", "Medicated for Autism", "Receives Behavioral Treatment for Autism", "Diagnosed with Behavior Problems", "Currently has Behavior Problems", "Severity of Behavior Problems", "Diagnosed with Depression", "Currently has Depression", "Severity of Depression", "Diagnosed with a Developmental Delay", "Currently has a Developmental Delay", "Severity of Developmental Delay", "Diagnosed with Down Syndrome", "Takes Emotion/Concentration/Behavior Medication", "Diagnosed with an Intellectual Disability", "Currently has an Intellectual Disability", "Severity of Intellectual Disability", "Diagnosed with a Learning Disability", "Currently has a Learning Disability", "Severity of Learning Disability", "Diagnosed with a Speech Disorder", "Currently has a Speech Disorder", "Severity of Speech Disorder", "Diagnosed with Tourette Syndrome", "Currently has Tourette Syndrome", "Severity of Tourette Syndrome"]
    return selected_vars

"""
@app.callback(
    Output("variables3", "value"),
    [Input("all-or-none3", "value")],
    [State("variables3", "options")],
)
def select_vars3(all_selected, options):
    selected_vars = []
    if all_selected:
        selected_vars = [option["value"] for option in options]
    else:
        selected_vars = ["Diagnosed with Autism", "Currently has Autism", "Severity of Autism", "Medicated for Autism", "Receives Behavioral Treatment for Autism", "Diagnosed with a Developmental Delay", "Currently has a Developmental Delay", "Severity of Developmental Delay", "Diagnosed with an Intellectual Disability", "Currently has an Intellectual Disability", "Severity of Intellectual Disability", "Diagnosed with a Learning Disability", "Currently has a Learning Disability", "Severity of Learning Disability", "Diagnosed with a Speech Disorder", "Currently has a Speech Disorder", "Severity of Speech Disorder"]
    return selected_vars
"""

if __name__ == '__main__':
    app.run_server(debug=True)