##############################################
# DATA ANALYSIS WITH PYTHON
##############################################
#- Numpy
#- Pandas
#- Data Visualization: Matplotlib & Seaborn
#- Advanced Functional Exploratory Data Analysis

###################################################
#-Why Numpy?
#-Crating Nump Arrays
#-Attibutes of Numpy Arrays
#-Reshaping
#-Index Selection
#-Slicing
#-Fancy Index
#-Conditions on Numpy
#-Mathematical Operations

###################################################
#- Why Numpy?
###################################################
import numpy as np
a=[1,2,3,4]
b=[2,3,4,5]

#-Classic python way

ab=[]

for i in range(0,len(a)):
     ab.append(a[i]* b[i])

# For numerical operations and computational operations with Numpy.
# It is preferred over lists because it is fast and keeps constant data.
# It enables operations to be performed at a higher level. It performs functional and vector operations.

a=np.array([1,2,3,4])
b=np.array([2,3,4,5])
a*b

####################
# Creating Numpy Arrays
#######################

import numpy as np

np.array([1,2,3,4,5])
type(np.array([1,2,3,4,5]))

np.zeros(10, dtype=int)
np.random.randint(0,10, size=10)
np.random.normal(10, 4, (3,4))

#######################
# Numpy Array Features(Attibutes of Numpy Arrays)
#######################

import numpy as np

# ndim: number of dimensions
# shape: size information
# size: total number of elements
# dtype: array data type

a=np.random.randint(10, size=5)

a.ndim
a.shape
a.size
a.dtype

####################
# Reshaping
####################
import numpy as np

np.random.randint(1, 10, size=9)
np.random.randint(1, 10, size=9).reshape(3,3)

ar= np.random.randint(1, 10, size=9)
ar.reshape(3,3)

####################
#- Index Selection
#################

import numpy as np
a= np.random.randint(10, size=10)
a[0]

a[0:5]

a[0]=999
a

m= np.random.randint(10, size=(3,5))

  m[0, 0]

  m[1,1]

  m[2,3]= 999
m
m[2,3]= 2.9
m

m[:,0]
m[1,:]
m[0:2, 0:3]

#################
# FancyIndex
#################

import numpy as np

v=np.arange(0,30, 3)
v[1]
v[4]

catch=[1,2,3]
v[catch]

#################
# Conditional Operations in Numpy
#################

import numpy as np
v= np.array([1,2,3,4,5])
v

####################
# Classic Loop
####################

ab= []
for i in v:
     if i < 3:
         ab.append(i)
#################
# with numpy
#################

v<3

v[v < 3]

v[v > 3]

v[v != 3]

v[v == 3]

v[v >= 3]


####################
# Mathematical Operations
#######################
import numpy as np
v= np.array([1, 2, 3, 4, 5,])

v/5
v*5/10
v**2
v- 1

np.subtract(v,1)
np.add(v, 1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)
v= np.subtract(v,1)

####################
# Solving Equations with Two Unknowns with Numpy
####################

# 5*x0 + x1=12
#x0 + 3*x1=10

a= np.array([[5,1], [1, 3]])
b= np.array([12,10])

np.linalg.solve(a,b)

#######################
#PANDAS
#######################

# PandasSeries
# Reading Data
# Quick Look at Data
# Selection Operations in Pandas
# Aggregation & Grouping
# Apply and Lambda
# Join Operations

#######################
# PandasSeries
#######################

import pandas as pd

s=pd.Series([10, 77, 12, 4, 5])
type(s)
s.index
s.dtype
s.size
s.ndim
s.values
type(s.values)
s.head(3)
s.tail(3)

####################
# Reading Data
####################

import pandas as pd
df = pd.read_csv("datasets/advertising.csv")
df.head()

# pandas cheatsheet

####################
# Quick Look at Data
####################
import pandas as pd
import seaborn as sns

df= sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()

#######################
# Selection operations in Pandas
#######################

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]

df.drop(0, axis=0).head()

delete_indexes=[1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)

# df= df.drop(delete_indexes, axis=0)
# df.drop(delete_indexes, axis=0, inplace=True)

##############
# Converting Variable to Index
#############################

df["age"].head()
df.age.head()

df.index = df["age"]

df.drop("age", axis=1).head()

df.drop("age", axis=1, inplace=True)
df.head(9)

##############
# Converting Index to Variable
#############################

df.index
df["age"]

df["age"]=df.index
df.head()
df.drop("age", axis=1, inplace=True)

df.reset_index().head()
df= df.reset_index()
df.head()

#################
# Operations on Variables
#############################

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"].head()

df.age.head()
type(df.age.head())

df[["age"]].head()
type(df[["age"]].head())

df[["age","alive"]]

col_names=["age", "adult_male", "alive"]
df[col_names]

df["age2"]= df["age"]**2
df["age3"]= df["age"]/ df["age2"]
df.head()

df.drop("age3", axis=1).head()

df.drop(col_names, axis=1).head()

df.loc[:, df.columns.str.contains("age")].head()

df.loc[:, ~df.columns.str.contains("age")].head()

####################### #
#iloc&loc
#######################


import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# iloc:integer based selection

df.iloc[0:3]

df.iloc[0, 0]

# loc: label based selection

df.loc[0:3]

df.iloc[0:3, "age"]
df.iloc[0:3, 0:3]

df.loc[0:3, "age"]

col_names=["age", "embarked", "alive"]
df.loc[0:3, col_names]

#######################
# Conditional Selection
#######################

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50].head()
df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, ["age", "class"]].head()

#Multiple conditions must be enclosed in parentheses
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "class"]].head()

df_new = df.loc[(df["age"] > 50) & (df["sex"] == "male")
        & ((df["embark_town"] == "Cherbourg")| (df["embark_town"] == "Southampton")),
        ["age", "class", "embark_town"]]

df_new["embark_town"].value_counts()

#######################
# Aggregation & Grouping
#######################

# - count()
# -first()
# - last()
# - mean()
# - median()
# - min()
# - max()
# - std()
# - there is()
# - sum()
# - pivot table


import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# Age variable mean
df["age"].mean()

# Average age by gender

df.groupby("sex")["age"].mean()

# Average and total age by gender

df.groupby("sex").agg({"age":"mean"})

# More agg functions for the age variable. If we want to apply it. We took the "sum" of the age variable.

df.groupby("sex").agg({"age":["mean","sum"]})

# Frequency related to the variable representing the port of embarkation after breaking it down by gender.

df.groupby("sex").agg({"age":["mean","sum"],
                        "embark_town":"count"})

# Depending on whether the mean of the Survived variable is 0 or 1.
# (74% of the women who boarded this ship survived, while 18% of the men survived)

df.groupby("sex").agg({"age":["mean","sum"],
                        "survived": ["mean"]})

# Breakdown by other categorical variables Breakdown by gender
# Then, let's take the average of the age and the survived variable after breaking down according to the port of embarkation.

# We prefer the two-level groupby list form.

df.groupby(["sex","embark_town"]).agg({"age":["mean"],
                        "survived": "mean"})

# Expansion by adding more classes. Adding a dimension with the Class expression, with 3 breakdowns.


df.groupby(["sex","embark_town","class"]).agg({"age":["mean"],
                        "survived": "mean"})

# Survival status according to frequency information.

df.groupby(["sex","embark_town","class"]).agg({
     "age":["mean"],
     "survived": "mean",
     "sex":"count"})

#######################
# Pivot Table
#######################

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# Reaching the age or survivor variable at the intersection of two variables, age and embarkation location.
# (default value of pivot table is mean)

df.pivot_table("survived","sex","embarked")

df.pivot_table("survived","sex","embarked", aggfunc="std")

# Adding more size information

df.pivot_table("survived","sex",["embarked","class"])

# Evaluation can be made according to a two-level index from both rows and columns.

df.pivot_table("survived",["sex","alive"],["embarked","class"])

# Breakdown by gender and by embarkation location or age. The age variable is a numerical variable.
# We turn the age variable into a categorical variable and look at its effect here.
# Cut function: To turn your numerical variables into categorical variables,
# ---If we know which categories we want to divide the numerical variable into, we use the "cut" function.
# ---If we do not know the numerical variable we have and want it to be divided into quarter values, the "qcut" function is used.
# 0-10: child 10-18: young 18 -28: young to older adults I define the mature age variable. In this cut function.
# For data that we cannot define, with the qcut function it automatically divides the values into categories according to the percentage and quarter values in the order from smallest to largest.

df.head()
df["new_age"] = pd.cut(df["age"],[0, 10, 18, 25, 40, 90])

# Survival rates by age and gender.

  df.pivot_table("survived", "sex", "new_age")


# Let's add the categorical version of the trip class.

df.pivot_table("survived", "sex", ["new_age","class"])

#Terminal display setting
# pd.set_option('display.width',500)

#######################
# Apply and Lambda
#######################

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()

# Apply--allows automatic function execution on rows or columns
# Lambda is a function definition method, the difference is its disposable feature:

# Let's create two new variables

df["age2"]= df["age"]*2
df["age3"]= df["age"]*5

# divide the age variables in this data set by 10

(df["age"]/10).head()
(df["age2"]/10).head()
(df["age3"]/10).head()

# Let's create it with a loop. We want to apply functions to the variables.

for col in df.columns:
     if "age" in col:
         print(col)

# Divide all values by 10

for col in df.columns:
     if "age" in col:
         print((df[col]/10).head())

# We have printed, now let's save it

for col in df.columns:
     if "age" in col:
         df[col]=df[col]/10
df.head()

# How do we do it with apply and lambda.
df[["age", "age2", "age3"]].apply(lambda x: x/10).head()

# Let's do it more programmatically

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x/10).head()

# With more pragmatic correspondence. An application that standardizes the values in the df to which it is applied

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x-x.mean())/ x.std()).head() # x value is variable.

# Using the function defined externally with def

def standard_scaler(col_name):
     return(col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standard_scaler).head()

With # Apply, we can use not only lambda but also our other standard functions.

# Apply func. It provides the opportunity to apply a certain function in rows or columns.

# Let's save the result of the operation

df.loc[:, ["age", "age2", "age3"]] =df.loc[:,df.columns.str.contains("age")].apply(standard_scaler).head()

# More automatic serial version

df.loc[:, df.columns.str.contains("age") ] =df.loc[:,df.columns.str.contains("age")].apply(standard_scaler).head()

df.head()

#######################
# Join Operations
#######################

import numpy as np
import pandas as pd
m= np.random.randint(1, 30, size=(5, 3))
df1= pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2= df1 + 99

# We merge two dataframes.
  pd.concat([df1, df2])

# Fix for index problem
pd.concat([df1, df2], ignore_index=True)

# This concatenation can also be done on a column basis by giving arguments to Concat

#######################
# Merging Operations with Merge
#######################

# More detailed merging operations can be done.

df1 = pd.DataFrame({'Employees':['john', 'dennis', 'mark', 'maria'],
                     'Group':['accounting', 'engineering', 'engineering', 'hr']})

df2= pd.DataFrame({'Employees':['john', 'dennis', 'mark', 'maria'],
                    'Start_date':[2010, 2009, 2014, 2019]})
# There are 5 different data structures. Df dictionary list integer string
# Each employee must have the same starting date

pd.merge(df1, df2)

# If we would like to specifically point out that it is for employees

pd.merge(df1, df2, on="Employees")

# We want to access the information of each employee's manager.

df3= pd.merge(df1, df2)

df4 = pd.DataFrame({'Group':['accounting','engineering','hr'],
                     'Manager':['Caner', 'Mustafa', 'Berkcan']})

# Let's join both tables by group

pd.merge(df3, df4)

####################### ########
# DATA VISUALIZATION: MATPLOTLIB & SEABORN
####################### ########

#######################
# MATPLOTLIB
#######################

# If there is a categorical variable: We visualize it with a column chart. With the countplot bar in Seaborn
# If there is a numerical variable: Histogram and Boxplot are used.
# Python is not the ideal place for data visualization..like Powerbi,clickwiev..

#######################
# Categorical Variable Visualization
#######################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()

# Let's choose a categorical variable and make it

df["sex"].value_counts().plot(kind='bar')
plt.show()

#######################
# Numerical Variable Visualization
#######################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()

# Historgram example age distribution information. (More people between the ages of 20-30 in the chart)

plt.hist(df["age"])
plt.show()

plt.hist(df["survived"])
plt.show()

plt.hist(df["pclass"])
plt.show()

# Boxplot with a box plot (can show outliers in the data set over quarters).

plt.boxplot(df["mouse"])
plt.show()

#######################
# Features of Matplotlib
#######################

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()

#################
# Plot
#################

x = np.array([1,8])
y = np.array([0,150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y, 'o')
plt.show()

####################
# Marker
#################

y = np.array([13, 28, 11, 100])

plt.plot(y, marker='o')
plt.show()

plt.plot(y, marker='*')
plt.show()

#############################
# Line
##############

y = np.array([13, 28, 11, 100])
plt.plot(y, linestyle="dashdot", color="r")
plt.show()
#############################
# Multiple Lines
##############

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

#############################
# Labels
##############

x= np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y= np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x, y)
plt.show()
# Title
plt.title("This Main Title")

# X axis naming
plt.xlabel("X axis naming")

# Y axis naming
plt.ylabel("Y axis naming")

# put grid
plt.grid()

##########################
# Subplots
#######################
# Displaying multiple images

# plot 1

x= np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y= np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 1)
plt.title("1")
plt.plot(x, y)


# plot 2

x= np.array([8, 8, 9, 9, 10, 15, 11, 15, 12, 15])
y= np.array([24, 20, 26, 27, 280, 29, 30, 30, 32, 33])
plt.subplot(1, 3, 2)
plt.title("2")
plt.plot(x, y)


# plot 3

x= np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y= np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 3, 3)
plt.title("3")
plt.plot(x, y)
plt.show()

#######################
# SEABORN VISUALIZATION
#######################

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

# Let's look at the class variables of the #gender variable

df["sex"].value_counts()

# To visualize--For categorical variables

sns.countplot(x=df["sex"], data=df)
plt.show()

# let's do it with matplot

df["sex"].value_counts().plot(kind='bar')
plt.show()

####################
# Numerical Variable Visualization
#######################

sns.boxplot(x=df["total_bill"])
plt.show()

# With Pandas function
df["total_bill"].hist()
plt.show()
sns.countplot



