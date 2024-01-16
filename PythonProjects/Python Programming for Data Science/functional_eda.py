###################################
# ADVANCED FUNCTIONAL EDA
#################################
# 1. General Situation
# 2. Analysis of Categorical Variables
# 3. Analysis of Numerical Variables
# 4. Analysis of Target Variable
# 5. Analysis of Correlation

#################################
# 1.General Situation
###############################
from typing import List, Any

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()

# Functions to be applied when we first receive the data set
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head=5):
    print("############################ Shape ###############")
    print(dataframe.shape)
    print("############################ Types ###############")
    print(dataframe.dtypes)
    print("############################ Head ###############")
    print(dataframe.head(head))
    print("############################ Tail ###############")
    print(dataframe.tail(head))
    print("############################ NA ###############")
    print(dataframe.isnull().sum())
    print("############################ Quantiles ###############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99,1]).T)

check_df(df)

# Let's try the function on a new dataset

df= sns.load_dataset("flights")
check_df(df)

#################################
# 2.Analysis of Categorical Variables
###############################

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()

# When there are too many variables;
# Capturing variable types with generalizability concerns
# And let's write functions that can analyze them specifically.

# If we want to analyze a single categorical variable

df["embarked"].value_counts()

# For unique values of another variable

df["sex"].unique()

# How many unique values are there in total

df["sex"].nunique()
df["class"].nunique()

# Automatically select all possible categorical variables from the data set
# First, according to the type information, we will then capture the values that are actually categorical but appear not to be.

df.info()

# To capture Categorical Variables according to type information;

cat_cols = [col for col in df.columns if str(df[col].dtypes)in ["category", "object", "bool"]]

# Capturing the Survived variable with a method (because it is actually a categorical variable

num_but_cat= [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

# Variables with high cardinality mean that they have too many classes to carry explanability value.
# To be able to programmatically capture non-categorical variables even though they are of categorical type;

cat_but_car= [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object" ]]

# All cat_cols will co-exist, so object bools, including synsirellas, are all in the same place.

cat_cols = cat_cols + num_but_cat

# What would we do if the variable came from cat_but_car?

cat_cols= [col for col in cat_cols if col not in cat_but_car ]

df[cat_cols]

# The consistency of the variables we choose to verify the transaction made

df[cat_cols].nunique()

# For numerals, let's select those that are not in cat_cols

[col for col in df.columns if col not in cat_cols]

# Let's write a function that we will evaluate functionally.
# Let this function get the value_counts of the values given to it (how many from which class are there)
# Let's print the percentage information of the classes.

df["survived"].value_counts() / len(df)
100 * df["survived"].value_counts() / len(df)

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")

# When we have extra variables (All categorical variables in the data set will be written automatically)

for col in cat_cols:
    cat_summary(df, col)


# cat summary funk if we want to add graphics

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)

# After the visual feature is added, let's try to send the cat summary function to scroll through all the variables.
# We must specify and separate this for different types, that's why there is an if structure in the loop


for col in cat_cols:
    if df[col].dtypes=="bool":
        print("sadasdasdasda")
    else:
    cat_summary(df, col, plot=True)

# If we want to change the "bool" type, we use the "adult-male" variable (true:1 false:0)
# We can do it because we know only one variable.

df["adult_male"].astype(int)

# If there is more than one and we do not know, redirect the variables with a loop

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col]=df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
    cat_summary(df, col, plot=True)

# Recommended first way to go (example scenario below):
# Instead of the loop operation outside, let's make the Type query inside and format it inside if the condition is not met.

def cat_summary(dataframe, col_name, plot=False):

    if dataframe[col_name].dtypes == "bool":
       dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
        else:
          print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
          print("##########################################")
        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

cat_summary(df,"adult_male", plot=True)

#################################
# 3. Analysis of Numerical Variables
###############################

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head()

# We want to examine the age and wage variable to access summary statistics in its simplest form.

df[["age","fare"]].describe().T

# let's bring cat_cols

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]


# How do we select numeric variables programmatically (with list comprehensions structure)

num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]

# Let's select variables that are in #num_cols and not in cat_cols

num_cols = [col for col in num_cols if col not in cat_cols]

# Let's write a function for these operations, we will just give df and the operation will return,
# We will write an analysis function. The important thing is to choose the data texts.

def num_summary(dataframe, numerical_col):
    # let's format the quarterly values.
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")

num_summary(df, "fare")

# Variables in case of more than one variable.

for col in num_cols:
    num_summary(df, col)

# Graphic display for num_summary or cat_summary function; by adding a property to this function.

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
num_summary(df, "age", plot=True)


# We have dozens of variables, how can we do it with them?

for col in num_cols:
    num_summary(df, col, plot=True)

#############################################
# Capturing Variables and Generalizing Operations.
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()
df.info()


# We must write a function that tells us;
# Fetch the categorical variable list and the numerical variable list separately.
# Additionally, give the categorical but cardinal list.
# (We are not interested in numerical-looking categoricals, we put them in cat_cols)

# We define the function docstring

def grab_col_names(dataframe, cat_th=10,  car_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.



    Parameters
    ----------
    dataframe: dataframe
         The variable names are the dataframe from which you want to retrieve them.
     cat_th: int, float
         It is the class threshold value for variables that are numeric but categorical.
     car_th: int, float
         is the class threshold value for categorical but cardinal variables.

    Returns
     -------
     cat_cols: list
         Categorical variable list
     num_cols: list
         Numerical variable list
     cat_but_car: list
         List of cardinal variables with categorical view

     Notes
     ------
     cat_cols + num_cols + cat_but_car = total number of variables
     num_but_cat is inside cat_cols.

    """
# The section that will create our categorical variables

     # cat_cols, cat_but_car(The section that will create the Categorical Variable List)
     # For cat_cols, first select ("category","object","bool")
     cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
     # Bringing those that are hidden from us such as "int or "float" and whose unique class number is less than 10
     num_but_cat = [col for col in dataframe.columns if
                    dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64"] ]
     # Bringing those whose types are "object" or "category" and whose number of unique classes is more than 20
     cat_but_car = [col for col in dataframe.columns if
                    dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object"]]
     # cat_cols customization
     cat_cols = cat_cols + num_but_cat
     # It was created to say that if something comes from cat_but_car, take the difference between the two.
     cat_cols = [col for col in cat_cols if col not in cat_but_car]

# The section that will create our numerical variables

     # We chose numeric variables by saying choose variables whose type is "int" or "float", but these should not be cat_cols.
     num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
     num_cols = [col for col in num_cols if col not in cat_cols]

     # For Reporting Statement

     print(f"Observations: {dataframe.shape[0]}")
     print(f"Variables: {dataframe.shape[1]}")
     print(f'cat_cols: {len(cat_cols)}')
     print(f'num_cols: {len(num_cols)}')
     print(f'cat_but_car: {len(cat_but_car)}')
     print(f'num_but_cat: {len(num_but_cat)}')

# To get lists out and not repeat. We keep the calculated values.

     return cat_cols, num_cols, cat_but_car

# I apply df and do not touch the values

cat_cols, num_cols, cat_but_car = grab_col_names(df)



def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

cat_summary(df, "sex")

for col in cat_cols:
    cat_summary(df, col)



def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

# BONUS ( cat_summary function. way to format it with the plot property)

df = sns.load_dataset("titanic")
df.info()

# Converting a specific data structure to a data structure we want

for col in df.columns:
     if df[col].dtypes =="bool":
         df[col] = df[col].astype(int)


cat_cols, num_cols, cat_but_car = grab_col_names(df

# We run cat_summary with graphics capability

def cat_summary(dataframe, col_name, plot=False):
     print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                         "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
     print("####################################################"

     if plot:
         sns.countplot(x=dataframe[col_name], data=dataframe)
         plt.show(block=True)

# Graph for categorical variables

for col in cat_cols:
  cat_summary(df, col, plot=True)

# Graphics for numeric variables
for col in num_cols:
     num_summary(df, col, plot=True)

#############################################
# 4. Analysis of Target Variable
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
     Note: Categorical variables with numerical appearance are also included.

     parameters
     ------
         dataframe: dataframe
                 Dataframe from which variable names are to be taken
         cat_th: int, optional
                 Class threshold value for variables that are numeric but categorical.
         car_th: int, optional
                 class threshold for categorical but cardinal variables.

     returns
     ------
         cat_cols: list
                 Categorical variable list.
         num_cols: list
                 Numerical variable list.
         cat_but_car: list
                 List of cardinal variables with categorical view.

     examples
     ------
         import seaborn as sns
         df = sns.load_dataset("iris")
         print(grab_col_names(df))


     Notes
     ------
         cat_cols + num_cols + cat_but_car = total number of variables
         num_but_cat is inside cat_cols.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

# To analyze the target variable we have in terms of categorical and numerical variables.
df["survived"].value_counts()
cat_summary(df, "survived")

# The question of what affects people's survival?
# Examining these variables should be done crosswise.
# In other words, we need to analyze by considering other variables according to the dependent variable.
# Multiplying the "survived" variable with categorical variables

#############################
# Analysis of Target Variable with Categorical Variables
#############################

df.groupby("sex")["survived"].mean()

# Target is the function that automatically processes the target variable with a categorical variable.

def target_summary_with_cat(dataframe, target, categorical_col):
     print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

target_summary_with_cat(df, "survived", "pclass")
target_summary_with_cat(df, "survived", "sex")

# Automatically browse the categorical variable list and evaluate the target quickly.

for col in cat_cols:
     target_summary_with_cat(df,"survived", col)

#############################
# Analysis of Target Variable with Numerical Variables
#############################

# As the opposite of the previous process, this time we take the target variable into groupby and get the average of the numerical variables.
# We bring the dependent variable to groupby and our numerical variable to aggretion.

df.groupby("survived")["age"].mean()

# Second way alternatively

df.groupby("survived").agg({"age":"mean"})

# Target is a function that automatically processes the target variable with a numeric variable.

def target_summary_with_num(dataframe, target, numerical_col):
     print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_num(df,"survived","age")

# Automatically browse the list of categorical variables and evaluate the target quickly

for col in num_cols:
     target_summary_with_num(df,"survived",col)

#############################################
# 5. Analysis of Correlation
#############################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/breast_cancer.csv")
df = df.iloc[:, 1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

# Calculation of Correlation

corr= df[num_cols].corr()

# Let's create a heat map.

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

##########################
# Deleting Highly Correlated Variables
##########################


cor_matrix = df.corr().abs()

# We should make the repeating variables in this matrix the same as in our own set (get rid of the repeating ones)

#0 1 2 3
# 0 1.000000 0.117570 0.871754 0.817941
#1 0.117570 1.000000 0.428440 0.366126
#2 0.871754 0.428440 1.000000 0.962865
#3 0.817941 0.366126 0.962865 1.000000


#0 1 2 3
#0 NaN 0.11757 0.871754 0.817941
#1 NaN NaN 0.428440 0.366126
#2 NaN NaN NaN 0.962865
#3 NaN NaN NaN NaN


upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90) ]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)

# Let's turn it into a function - it is very important to be able to separate variables that are related to each other.

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
     corr = dataframe.corr()
     cor_matrix = corr.abs()
     upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
     drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
     if plot:
         import seaborn as sns
         import matplotlib.pyplot as plt
         sns.set(rc={'figure.figsize': (15, 15)})
         sns.heatmap(corr, cmap="RdBu")
         plt.show()
     return drop_list

high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)