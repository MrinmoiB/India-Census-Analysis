#!/usr/bin/env python
# coding: utf-8

# In[15]:



import pandas as pd
from itertools import groupby
from __future__ import division
import operator
import matplotlib.pyplot as plt
import matplotlib.cm

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import numpy as np
import seaborn as sns
import math


# In[2]:



data = pd.read_csv("india-districts-census-2011.csv")


# In[3]:



data.shape


# In[4]:



data.head()


# In[5]:



data.describe()


# In[6]:



print(data.groupby('State name').size())


# In[7]:


'''
Calculating state wise literacy rates
'''


states_group = data.groupby(by = "State name")

literacy_rate = []


for key , group in states_group:
    total_state_pop = 0
    total_literate_pop = 0
    for row in group.iterrows():
        total_state_pop += row[1][3] 
        total_literate_pop += row[1][6]
    
    rate = (total_literate_pop/total_state_pop)*100
    literacy_rate.append((key,rate))
    
print ("Statewise literacy rates : \n")
print (literacy_rate)


# In[9]:


'''
STEP 2 : CREATING A MAP
'''
fig, ax = plt.subplots()
m = Basemap(projection='merc',lat_0=54.5, lon_0=-4.36,llcrnrlon=68.1, llcrnrlat= 6.5, urcrnrlon=97.4, urcrnrlat=35.5)
m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')
m.drawcoastlines()

'''
STEP 3 : USING SHAPEFILES FOR DRAWING STATES 
'''
m.readshapefile("D:\MSIT\gramener\INDIA","INDIA")


'''
STEP 4 : CREATING A DATAFRAME MAPPING SHAPES TO STATE NAME AND LITERACY RATES
'''
lit_rate = []
for state_info in m.INDIA_info:
    state = state_info['ST_NAME'].upper()
    rate = 0
    
    for x in literacy_rate:
        if x[0] == state:
            rate = x[1]
            break
    lit_rate.append(rate)            
    
df_poly = pd.DataFrame({
        'shapes': [Polygon(np.array(shape), True) for shape in m.INDIA],
        'area': [area['ST_NAME'] for area in m.INDIA_info],
        'lit_rate' : lit_rate
    })

'''
STEP 5 : USING DATA TO COLOR AREAS
'''
shapes = [Polygon(np.array(shape), True) for shape in m.INDIA]
cmap = plt.get_cmap('Oranges')   
pc = PatchCollection(shapes, zorder=2)

norm = Normalize()
pc.set_facecolor(cmap(norm(df_poly['lit_rate'].fillna(0).values)))
ax.add_collection(pc)

mapper = matplotlib.cm.ScalarMappable(cmap=cmap)
mapper.set_array(lit_rate)
plt.colorbar(mapper, shrink=0.4)
ax.set_title("LITERACY RATES OF INDIAN STATES")
plt.rcParams['figure.figsize'] = (15,15)
plt.rcParams.update({'font.size': 20})
plt.show()


# In[10]:


literacy_rate.sort(key = operator.itemgetter(1))
print literacy_rate[:5]


#  Find out most similar districts in Bihar and Tamil Nadu. Similarity can be based on any of the columns from the data.

# In[11]:


df_Bihar = data.loc[data['State name'] == 'BIHAR']
df_Tamil_Nadu = data.loc[data['State name'] == 'TAMIL NADU']


# The Bihar dataframe :

# In[12]:


df_Bihar.head()


# 
# 
# 
# And, for Tamil Nadu :

# In[13]:


print "rows = %s"%(str(len(df_Tamil_Nadu)))
df_Tamil_Nadu.head()


# In[31]:


def similar_districts(df1, df2):
    df1.set_index('District code')
    df2.set_index('District code')
    
    main_diff = []
    
    for row1 in df1.iterrows(): 
        diff=[]
        for row2 in df2.iterrows():
            dist = 0
            for column in list(data)[3:]:
                max_col = max(data[column])
                min_col = min(data[column]) 
    
                dist += pow((row1[1][column] - row2[1][column])/(max_col - min_col),2)
            diff.append(1/math.sqrt(dist))
        main_diff.append(diff)

    max_val = 0
    max_index1 = 0
    max_index2 = 0
    for i in range(len(main_diff)):

        for j in range(len(main_diff[i])):
            if(main_diff[i][j] > max_val):
                max_val = main_diff[i][j]
                max_index1 = i
                max_index2 = j

    print "%s from Bihar and %s from Tamil Nadu are most similar" %(df1['District name'].iloc[max_index1],
                                                                    df2['District name'].iloc[max_index2])
    return main_diff
    
sim_matrix = similar_districts(df_Bihar, df_Tamil_Nadu)


# In[37]:


norm=Normalize()
ax = plt.axes()
sns.heatmap(norm(sim_matrix), xticklabels=df_Tamil_Nadu['District name'],yticklabels=df_Bihar['District name'],
            linewidths=0.05,cmap='Blues').set_title("SIMILARITY MATRIX FOR DISTRICTS OF BIHAR AND TAMIL NADU")
plt.rcParams['figure.figsize'] = (15,15)


# ##  How does the mobile penetration vary in regions (districts or states) with high or low agricultural workers?
# 
# 

# In[47]:


states_group = data.groupby(by = "State name")

households_with_mobile = []
agri_workers = []

for key , group in states_group:
    total_mobi_pop = 0
    total_agri_workers = 0
    
    for row in group.iterrows():
    
        total_mobi_pop += row[1][59]
        total_agri_workers += row[1][22]
    
    households_with_mobile.append((key,total_mobi_pop))
    agri_workers.append((key,total_agri_workers))
    
df_mobile_penetration =  pd.DataFrame({
        'state' : [x[0] for x in households_with_mobile] ,
        'Households_with_mobile': [x[1] for x in households_with_mobile],
        'agri_workers' : [x[1] for x in agri_workers]
        
    })

df_mobile_penetration


# In[46]:


from numpy import *
ind = arange(35)
width = 0.4

fig, ax = plt.subplots()
plt.xlim(0,22000000)
rects1 = ax.barh(ind, df_mobile_penetration['agri_workers'],width,color='g',align='center')
rects2 = ax.barh(ind+width, df_mobile_penetration['Households_with_mobile'],width,color='b',align='center')
ax.set_xlabel('Population')
ax.set_title('MOBILE PENETRATION IN VARIOUS STATES W.R.T. AGRICULTURAL WORKERS')
ax.set_yticks(ind + width / 2)
ax.set_yticklabels((x for x in df_mobile_penetration['state']))
ax.legend((rects1[0], rects2[0]), ('Agricultural Workers', 'No. of households using mobiles'))
plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = (20,20)
plt.show()


# States like **Maharashtra** and **UP** have very less difference in number of agriclutural workers and number of households using mobiles.
# 
# States like **Bihar, AP and MP** have a significant amount of difference in number of agriclutural workers and number of households using mobiles.
