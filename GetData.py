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

### Screener Dataframes
df_s_2020 = pd.read_stata(screener_2020)
#df_s_2020.to_csv('./data/nsch_2020_screener.csv', index=False)

df_s_2019 = pd.read_stata(screener_2019)
#df_s_2019.to_csv('./data/nsch_2019_screener.csv', index=False)

df_s_2018 = pd.read_stata(screener_2018)
#df_s_2018.to_csv('./data/nsch_2018_screener.csv', index=False)

df_s_2017 = pd.read_stata(screener_2017)
#df_s_2017.to_csv('./data/nsch_2017_screener.csv', index=False)

df_s_2016 = pd.read_stata(screener_2016)
#df_s_2016.to_csv('./data/nsch_2016_screener.csv', index=False)

### Topical Dataframes
df_t_2020 = pd.read_stata(topical_2020)
#df_t_2020.to_csv('./data/nsch_2020_topical.csv', index=False)

df_t_2019 = pd.read_stata(topical_2019)
#df_t_2019.to_csv('./data/nsch_2019_topical.csv', index=False)

df_t_2018 = pd.read_stata(topical_2018)
#df_t_2018.to_csv('./data/nsch_2018_topical.csv', index=False)

df_t_2017 = pd.read_stata(topical_2017)
#df_t_2017.to_csv('./data/nsch_2017_topical.csv', index=False)

df_t_2016 = pd.read_stata(topical_2016)
#df_t_2016.to_csv('./data/nsch_2016_topical.csv', index=False)

print("done")