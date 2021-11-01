import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
plt.rcParams["figure.figsize"] = (20, 15)

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

### Create Correlation Table of All 2020 Data
#all_data = pd.read_csv('./data/nsch_2020_topical.csv')
#corr = selected_data.corr()
#corr.style.background_gradient(cmap='coolwarm').set_precision(4).to_excel('./corr_tables/nsch_2020_topical.xlsx', engine='xlsxwriter')

### Create Correlation Table of Only Selected Variables
#selected_data = pd.read_csv('./data/nsch_2020_topical.csv')[all_selected_vars]
#corr = selected_data.corr()
#corr.style.background_gradient(cmap='coolwarm').set_precision(4).to_excel('./corr_tables/nsch_2020_topical_selected_data.xlsx', engine='xlsxwriter')

### Create Correlation Table of Adult Info
#adult_data = pd.read_csv('./data/nsch_2020_topical.csv')[all_adult_vars]
#corr = adult_data.corr()
#corr.style.background_gradient(cmap='coolwarm').set_precision(4).to_excel('./corr_tables/nsch_2020_topical_adult_data.xlsx', engine='xlsxwriter')

### Create Correlation Table of Child Info
#child_data = pd.read_csv('./data/nsch_2020_topical.csv')[all_child_vars]
#corr = child_data.corr()
#corr.style.background_gradient(cmap='coolwarm').set_precision(4).to_excel('./corr_tables/nsch_2020_topical_child_data.xlsx', engine='xlsxwriter')

### Create Correlation Table of Child Health Info
#child_health_data = pd.read_csv('./data/nsch_2020_topical.csv')[all_child_health]
#corr = child_health_data.corr()
#corr.style.background_gradient(cmap='coolwarm').set_precision(4).to_excel('./corr_tables/nsch_2020_topical_child_health_data.xlsx', engine='xlsxwriter')

### Create Correlation Table of Columns of Interest
#data_of_interest = pd.read_csv('./data/nsch_2020_topical.csv')[columns_of_interest]
#corr = data_of_interest.corr()
#corr.style.background_gradient(cmap='coolwarm').set_precision(4).to_excel('./corr_tables/nsch_2020_topical_data_of_interest.xlsx', engine='xlsxwriter')

### Get Top Correlations
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

data_of_interest = pd.read_csv('./data/nsch_2020_topical.csv')[columns_of_interest]
print("Top Correlations")
print(get_top_abs_correlations(data_of_interest, 25))

corr = data_of_interest.corr()
filtered_corr = corr[((corr >= .5) | (corr <= -.5)) & (corr !=1.000)]
#filtered_corr.style.background_gradient(cmap='coolwarm').set_precision(4).to_excel('./corr_tables/nsch_2020_topical_data_of_interest_filtered.xlsx', engine='xlsxwriter')

sn.heatmap(filtered_corr, annot=False, cmap="coolwarm", linewidths=.5, linecolor='gray')
plt.savefig('./heatmaps/nsch_2020_topical_data_of_interest.png')