### Uncomment line below for interactive funtionality (only works in notebook or interactive window)
#%matplotlib widget
from ast import literal_eval
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

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

df = pd.read_csv('./data/nsch_2020_topical.csv')

x1 = df[adult1_columns]
x2 = df[adult2_columns]
x3 = df[gen_adult_info]
x4 = df[gen_child_info]
x5 = df[child_experiences]

y1 = df[ment_health]
y2 = df[phys_health]

x1_lst = list(x1.columns)
x2_lst = list(x2.columns)
x3_lst = list(x3.columns)
x4_lst = list(x4.columns)
x5_lst = list(x5.columns)

y1_lst = list(y1.columns)
y2_lst = list(y2.columns)

var_lsts = [y1_lst, x4_lst, x5_lst]

def get_top_multi_reg(var_lists):
    combos = []
    for element in itertools.product(*var_lists):
        combos.append(element)
    res_lst = []
    for combo in combos:
        if len(combo) <= 1:
            print("Combo too small")
        elif len(combo) == 2:
            print("Linear Regression")
            y1 = combo[0]
            Y = df[y1]
            Y = Y.fillna(0)
            
            x1 = combo[1]
            X = df[x1]
            X = X.fillna(0)

            X = sm.add_constant(X) # adding a constant
            model = sm.OLS(Y, X).fit()
            corr = model.rsquared

            result = ("Combo = ", str(combo), "R-Squared = ", corr)
            res_lst.append(result)
        elif len(combo) == 3:
            y1 = combo[0]
            Y = df[y1]
            Y = Y.fillna(0)
            
            x1 = combo[1]
            x2 = combo[2]
            X = df[[x1, x2]]
            X = X.fillna(0)

            X = sm.add_constant(X) # adding a constant
            model = sm.OLS(Y, X).fit()
            corr = model.rsquared

            result = ("Combo = ", str(combo), "R-Squared = ", corr)
            res_lst.append(result)
        elif len(combo) == 4:
            y1 = combo[0]
            Y = df[y1]
            Y = Y.fillna(0)
            
            x1 = combo[1]
            x2 = combo[2]
            x3 = combo[3]
            X = df[[x1, x2, x3]]
            X = X.fillna(0)

            X = sm.add_constant(X) # adding a constant
            model = sm.OLS(Y, X).fit()
            corr = model.rsquared

            result = ("Combo = ", str(combo), "R-Squared = ", corr)
            res_lst.append(result)
        else:
            print("Combo too big")
    top_res = sorted(res_lst, key=lambda x: x[3], reverse=True)[:5]
    return top_res

#get_top_multi_reg(var_lsts)

def graph_top_multi_reg(var_lists):
    top_res = get_top_multi_reg(var_lists)
    top_combo = top_res[0][1]
    top_combo = literal_eval(top_combo)
    print(top_combo)
    if len(top_combo) <= 1:
        print("Combo to small")
    elif len(top_combo) == 2:
        print("Linear Regression")
        y1 = top_combo[0]
        Y = df[y1]
        Y = Y.fillna(0)
        
        x1 = top_combo[1]
        X = df[x1]
        X = X.fillna(0)

        plt.plot(X, Y, 'o')
        plt.xlabel(x1)
        plt.ylabel(y1)
        m, b = np.polyfit(X, Y, 1)
        plt.plot(X, m*X + b)
        plt.show()
    elif len(top_combo) == 3:
        y1 = top_combo[0]
        Y = df[y1]
        Y = Y.fillna(0)
        
        x1 = top_combo[1]
        x2 = top_combo[2]
        X1 = df[x1]
        X1 = X1.fillna(0)
        X2 = df[x2]
        X2 = X2.fillna(0)
        X = df[[x1,x2]]
        X = X.fillna(0)

        #"""
        ### Strat 1
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X1, X2, Y, marker='.', color='red')
        ax.set_xlabel(x1)
        ax.set_ylabel(x2)
        ax.set_zlabel(y1)
        
        model = linear_model.LinearRegression()
        model.fit(X, Y)
        coefs = model.coef_
        intercept = model.intercept_
        xs = np.tile(np.arange(4), (4,1))
        ys = np.tile(np.arange(4), (4,1)).T
        zs = xs*coefs[0]+ys*coefs[1]+intercept

        ax.plot_surface(xs,ys,zs, alpha=0.5)
        plt.show()
        #"""

        """ 
        ### Strat 2
        fig = plt.figure()
        ax1 = fig.add_subplot(131, projection='3d')

        x1_pred = np.linspace(0, 3, 20)    
        x2_pred = np.linspace(0, 3, 20) 
        xx1_pred, xx2_pred = np.meshgrid(x1_pred, x2_pred)
        model_viz = np.array([xx1_pred.flatten(), xx2_pred.flatten()]).T
        ols = linear_model.LinearRegression()
        model = ols.fit(X, Y)
        predicted = model.predict(model_viz)
        r2 = model.score(X, Y)

        ax1.plot(X1, X2, Y, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax1.scatter(xx1_pred.flatten(), xx2_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
        ax1.set_xlabel(x1, fontsize=12)
        ax1.set_ylabel(x2, fontsize=12)
        ax1.set_zlabel(y1, fontsize=12)
        ax1.locator_params(nbins=4, axis='x')
        ax1.locator_params(nbins=5, axis='x')

        ax1.view_init(elev=27, azim=112)

        fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)

        fig.show()
        """

    elif len(top_combo) >= 4:
        print("Combo too big for graphing")

graph_top_multi_reg(var_lsts)