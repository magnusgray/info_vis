import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

### Screener Files
screener_2020 = './data/nsch_2020_screener.dta'
screener_2019 = './data/nsch_2019_screener.dta'
screener_2018 = './data/nsch_2018_screener.dta'
screener_2017 = './data/nsch_2017_screener.dta'
screener_2016 = './data/nsch_2016_screener.dta'

### Topical Files
topical_2020 = './data/nsch_2020_topical.dta'
topical_2019 = './data/nsch_2019_topical.dta'
topical_2018 = './data/nsch_2018_topical.dta'
topical_2017 = './data/nsch_2017_topical.dta'
topical_2016 = './data/nsch_2016_topical.dta'

### For random test
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

columns_of_interest = ['sc_age_years', 'sc_race_r', 'sc_sex',] + ['k2q35a', 'k2q35b', 'k2q35c', 'autismmed', 'autismtreat']
labels =  ["Age", "Race", "Sex", "Diagnosed_with_Autism", "Currently_has_Autism", "Severity__of_Autism", "Medicated_for_Autism", "Behavioral_Treatment_for_Autism"]
df = pd.read_stata(topical_2020)[columns_of_interest]
df.columns = labels
data = df
data = data.replace({'Race': {1:"White", 2:"Black/African American", 3:"American Indian or Alaskan Native", 4:"Asian", 5:"Native Hawaiian or Other Pacific Islander", 7:"Two or More Races"},
'Sex': {1:"Male", 2:"Female"}, 'Diagnosed_with_Autism': {1:"Yes", 2:"No"},
'Currently_has_Autism': {1:"Yes", 2:"No"}, "Severity_of_Autism": {1:"Mild", 2:"Moderate", 3:"Severe"},
'Medicated_for_Autism': {1:"Yes", 2:"No"}, 'Behavioral_Treatment_for_Autism': {1:"Yes", 2:"No"}})
data = data.dropna()
data.to_csv('./data/nsch_2020_topical_reduced_newest.csv', index=False)

### Screener Dataframes
#df_s_2020 = pd.read_stata(screener_2020)
#df_s_2020.to_csv('./data/nsch_2020_screener.csv', index=False)

#df_s_2019 = pd.read_stata(screener_2019)
#df_s_2019.to_csv('./data/nsch_2019_screener.csv', index=False)

#df_s_2018 = pd.read_stata(screener_2018)
#df_s_2018.to_csv('./data/nsch_2018_screener.csv', index=False)

#df_s_2017 = pd.read_stata(screener_2017)
#df_s_2017.to_csv('./data/nsch_2017_screener.csv', index=False)

#df_s_2016 = pd.read_stata(screener_2016)
#df_s_2016.to_csv('./data/nsch_2016_screener.csv', index=False)

### Topical Dataframes
#df_t_2020 = pd.read_stata(topical_2020)
#df_t_2020.to_csv('./data/nsch_2020_topical.csv', index=False)

#df_t_2019 = pd.read_stata(topical_2019)
#df_t_2019.to_csv('./data/nsch_2019_topical.csv', index=False)

#df_t_2018 = pd.read_stata(topical_2018)
#df_t_2018.to_csv('./data/nsch_2018_topical.csv', index=False)

#df_t_2017 = pd.read_stata(topical_2017)
#df_t_2017.to_csv('./data/nsch_2017_topical.csv', index=False)

#df_t_2016 = pd.read_stata(topical_2016)
#df_t_2016.to_csv('./data/nsch_2016_topical.csv', index=False)

print("done")