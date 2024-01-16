#################
# Adding Properties and Docstrings to Functions
#################



#############################
# Task 1: Adding Properties to Functions
#############################

# Task: Add 1 property to the cat_summary() function. Let this property be formatted with arguments.
# You can also make an existing feature controllable with an argument.

# What does it mean to add a property that can be formatted with an argument to the function?
# For example, 2 properties that can be formatted with arguments have been added to the check_df function below.
# With these properties, how many observations the tail function will display and whether quantile values will be displayed are entered as properties in the function.
# The user can format these properties with arguments.


df=sns.load_dataset('titanic')
df.head()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_df(dataframe, head=5, tail=5, quan=True):
     print("################################## Shape #################")
     print(dataframe.shape)
     print("################################## Types #################")
     print(dataframe.dtypes)
     print("################################## Head ##################")
     print(dataframe.head(head))
     print("################################## Tail ################")
     print(dataframe.tail(head))
     print("##################################NA ##################")
     print(dataframe.isnull().sum())
     if quan:
        print("################################## Quantiles #################")
        print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99,1]).T)

check_df(df)

check_df(df, head=3, tail=3, quan=True)

def cat_summary(dataframe, col_name, plot=False):
     print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                         "Ratio":100 * dataframe[col_name].value_counts() / len(dataframe)}))
     print("####################################################"

     if plot:
         sns.countplot(x=dataframe[col_name], data=dataframe)
         plt.show(block=True)

cat_summary(df, "sex", plot=True)




def cat_summary(dataframe, col_name, plot=False, ratio=True):
     if ratio:
         print(pd.DataFrame({col_name:dataframe[col_name].value_counts(),
                             "Ratio": 100 * dataframe[col_name].value_counts()/len(dataframe)}))
         print("########################################"
     else:
         print(pd.DataFrame({col_name: dataframe[col_name].value_counts()}))
         print("########################################"
     if plot:
         sns.countplot(x=dataframe[col_name], data=dataframe)
         plt.show()

cat_summary(df, "sex", plot=True, ratio=True)

#################
# Task 2: Docstring Writing
#############################

# Write a numpy style docstring containing 4 pieces of information (if appropriate) for the check_df(), cat_summary() functions. (task, params, return, example)

def check_df(dataframe, head=5):
     """

     It gives information about the shape of the dataframe,type of its variables,first 5 observations,last 5 observations.Total number of missing observations and quartiles

      parameters
      ------
          dataframe: dataframe
                  The dataframe whose properties will be defined

          head: int, optional
                  References to number of observations that will be displayed from the beginning and end

      returns
      ------
      None

      examples
      ------
         check_df(df,7)


      Notes
      ------
      The number of observations to be viewed from the end is the some as the number of observations to be observed from the beginning
     """
     print("################################## Shape #################")
     print(dataframe.shape)
     print("################################## Types #################")
     print(dataframe.dtypes)
     print("################################## Head ##################")
     print(dataframe.head(head))
     print("################################## Tail ################")
     print(dataframe.tail(head))
     print("##################################NA ##################")
     print(dataframe.isnull().sum())


  def cat_summary(dataframe, col_name, plot=False):
  """
  It gives the percantage calculation of the category in the selected variable relative to the entire dataframe.
  Note: The selected variable must be a categorical variable.

      parameters
      ------
          dataframe: dataframe
                  The dataframe whose percentage of variables will be calculated

          col_name: [str]
                   The name of the variable which is calculated as a percentile
          plot: bool, optional
                    It decides whether the data results are to be visualized or not.

      returns
      ------
      None

      examples
      ------
         import seaborn as sns
         df =sns.load_dataset("titanic")
         cat_summary(df,"Survived",plot=True)

  """""
          print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                             "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
         print("####################################################"

         if plot:
             sns.countplot(x=dataframe[col_name], data=dataframe)
             plt.show(block=True)





