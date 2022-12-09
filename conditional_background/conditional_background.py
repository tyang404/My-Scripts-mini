# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 19:09:56 2022

@author: yangt
"""

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.DataFrame({'myDatarange': np.random.randint(-150, -50, size=50)})
ranges = [-10**6, -140, -123, -110, -100, -90, 10**6]
df['data_classification'] = pd.cut(df['myDatarange'], ranges, right=False,
                                   labels=['Off', 'Bad', 'Poor', 'Moderate', 'Good', 'Very Good'])
df['myDatarange2'] = df['myDatarange']*2
fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df.index, df['myDatarange'], color='blue', linewidth=2, alpha=0.9, label="myDataRange")

sns.barplot(x=df.index, y=[df['myDatarange'].min()] * len(df),
            hue='data_classification', alpha=0.5, palette='inferno', dodge=False, data=df, ax=ax1)
for bar in ax1.patches: # optionally set the bars to fill the complete background, default seaborn sets the width to about 80%
    bar.set_width(1)

plt.legend(bbox_to_anchor=(1.02, 1.05) , loc='upper left')
plt.tight_layout()
plt.show()
