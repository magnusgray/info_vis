Information Visualization Group Project

Dataset: National Survey of Children's Health (NSCH)

Data Source: https://www.census.gov/programs-surveys/nsch/data/datasets.html

Notes to Self:
    Fill out requirements.txt : pip freeze > requirements.txt
    Install requiremnets.txt : pip install -r requirements.txt

Need to Know:
    
    Folders:
        causal_models : Contains images of causal models built for presentation

        corr_tables : Contains Excel files of the correlation tables made in QuickAnalysis.py

        data : Contains data in orginal format (.dta) from 2016-2020 as well as data in .csv format
            Also contains .txt file with variable information 

        heatmaps : Contains images of the Seaborn heatmaps created in QuickAnalysis.py

    Files:
        .gitignore : stuff to ignore when pushed to github

        app.py : the main app created for this project
            run in a terminal and click on the dash link to access the app on local server

        CauseEffect.py : file where the ADD causal model was created

        CauseEffect2.py : file where the Autism causal model was created

        GetData.py : file that converted .dta files to .csv files 

        Interactive.py : file for orginial interactive heatmap 
            run file in interactive window to see interactivity
            alternatively, transfer to a jupyter notebook for interactivity

        MultiReg.py : file for conducting multiple regression analysis
            contains a function that looks for highest R^2 value among sets or combinations of variables
            also contains a function that will graph top variable combo on interactive 3D plot
                run file in interactive window to see interactivity
                alternatively, transfer to a jupyter notebook for interactivity

        NodeLink.py : file used to build node link diagrams
            must downgrade to previous versions of some package as some functionality depreciated
            first stategy groups variables based on there correlations and creates a diagram for each group
            second strategy puts all variables into a circle
                then draws edges connecting positive or negatvive correlations based on which is selected
            
        Procfile : would have been used for Heroku app
            opted to instead host app on PythonAnywhere

        QuickAnalysis: file first used for analysis of dataset
            created correlation tables, found top correlations, and created heatmaps

        README.md : file you are reading right now

        requirements.txt : contains packages currently installed in virtual env

        Variables.docx : lists the variables I found to be interesting
            also provides a breif description of each variable

    General/Other: 
        Everything was done in Python and in VS Code
        Only the 2020 topical dataset was used in the analysis
        Of the original 400+ variables in the 2020 dataset, only a little over 100 were analyzed
        There were over 40,000 records in this dataset
        In some instances, NaN values had to be replaced with 0
        
