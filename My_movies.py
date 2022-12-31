# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:47:24 2022

@author: Rahul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Association Rules\\my_movies.csv")
df.shape
df.head()
df.info()
df1 = df.iloc[:,5:]
df1

# Apriori Algorithm
# 1. Association rules with 10% Support and 70% confidence

from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

# with 10% support
frequent_itemsets = apriori(df1,min_support = 0.1,use_colnames=True)
frequent_itemsets

# 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules

# Lift Ratio>1 is a good influential rule is selecting the associated
rules[rules.lift>1]

# Visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

# 2. Association rules with 5% Support and 90% confidence
# with 5% support
frequent_itemsets2=apriori(df1,min_support=0.05,use_colnames=True)
frequent_itemsets2

# 90% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.9)
rules2

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules2[rules2.lift>1]

# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


















