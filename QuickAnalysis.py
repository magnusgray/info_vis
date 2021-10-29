import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (100, 100)

data = pd.read_csv('./data/nsch_2020_topical.csv')

corr = data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(4).to_excel('./corr_tables/nsch_2020_topical.xlsx', engine='xlsxwriter')

#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
#fig.colorbar(cax)
#ticks = np.arange(0,len(data.columns),1)
#ax.set_xticks(ticks)
#plt.xticks(rotation=90)
#ax.set_yticks(ticks)
#ax.set_xticklabels(data.columns)
#ax.set_yticklabels(data.columns)
#plt.show()