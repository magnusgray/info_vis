import plotly.graph_objects as go
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

df = pd.read_csv('./data/nsch_2020_topical.csv')[columns_of_interest]
df.columns = labels
data = df

corr = data.corr()

### Strategy 1
"""
links = corr.stack().reset_index()
links.columns = ['var1', 'var2', 'value']
 
links_filtered=links.loc[ (abs(links['value']) > 0.5) & (links['var1'] != links['var2']) ]
 
G = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2', edge_attr=True)

fig = plt.figure("Node Link Diagram with Selected Variables",figsize=(30,30)) 
nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=10)
fig.show()
"""

### Strategy 2
"""
cor_matrix = np.asmatrix(corr)

G = nx.from_numpy_matrix(cor_matrix)
#G = nx.relabel_nodes(G,lambda x: columns_of_interest[x])
G.edges(data=True)

def create_corr_network(G, corr_direction, min_correlation):
    H = G.copy()
    
    for var1, var2, weight in G.edges(data=True):
        if corr_direction == "positive":
            if weight["weight"] <0 or weight["weight"] < min_correlation:
                H.remove_edge(var1, var2)
        else:
            if weight["weight"] >=0 or weight["weight"] > min_correlation:
                H.remove_edge(var1, var2)
                
    edges,weights = zip(*nx.get_edge_attributes(H,'weight').items())
    
    weights = tuple([(1+abs(x))**2 for x in weights])
    
    d = nx.degree(H)
    nodelist, node_sizes = zip(*d.items())

    positions=nx.circular_layout(H)
    
    plt.figure(figsize=(15,15))

    nx.draw_networkx_nodes(H,positions,node_color='#DA70D6',nodelist=nodelist, node_size=tuple([x**3 for x in node_sizes]),alpha=0.8)
    
    nx.draw_networkx_labels(H, positions, font_size=8, font_family='sans-serif')
    
    if corr_direction == "positive":
        edge_colour = plt.cm.GnBu 
    else:
        edge_colour = plt.cm.PuRd
        
    nx.draw_networkx_edges(H, positions, edge_list=edges,style='solid', width=weights, edge_color = weights, edge_cmap = edge_colour, edge_vmin = min(weights), edge_vmax=max(weights))

    plt.axis('off')
    #plt.savefig("" + corr_direction + ".png", format="PNG")
    plt.show() 

create_corr_network(G, corr_direction="positive", min_correlation = 0.5)

create_corr_network(G, corr_direction="negative", min_correlation = -0.5)
"""