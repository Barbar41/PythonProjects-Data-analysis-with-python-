####################
# Pandas Exercises
#################

import pandas as pd
import seaborn as sns
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)

# Task 1: Identify the Titanic dataset from the Seaborn library.

df = sns.load_dataset("titanic")
df.head()
df.shape
df.info()

# Task 2: Find the number of male and female passengers in the Titanic data set.

df["sex"].value_counts()

# Task 3: Find the number of unique values for each column.

df.nunique()

# Task 4: Find the number of unique values of the pclass variable.

df["pclass"].unique()

# Task 5: Find the number of unique values of the pclass and parch variables.

df[["pclass","parch"]].nunique()

# Task 6: Check the type of the embarked variable. Change its type to category and check again.

df["embarked"].dtype #object
df["embarked"]= df["embarked"].astype("category")
df["embarked"].dtype #category

# Task 7: Show all the wisdoms of those with embarked value C.

df[df["embarked"] == "C"].head(20)

# Task 8: Show all the wisdoms of those whose embarked value is not S.

df[df["embarked"] != "S"]["embarked"].unique()

# Task 9: Show all the information of passengers who are women and under 30 years old.

df[(df["age"]<30) & (df["sex"]=="female")].head()

# Task 10: Show the mouse the information of passengers older than 500 or older than 70 years old.

df[(df["mouse"] > 500) | (df["age"]<70)].head()

# Task 11: Find the sum of the null values in each variable.

df.isnull().sum()

# Task 12: Remove the who variable from the dataframe.

df.drop("who", axis=1, inplace=True)
df.head()
df.columns


# Task 13: Fill the empty values in the deck variable with the most repeated value (mode) of the deck variable.

deck_mode= df["deck"].mode()[0]
df["deck"].fillna(deck_mode, inplace=True)
df.head()

# Task 14: Fill the empty values in the age variable with the median of the age variable.
  #-method 1
age_med= df["age"].median()
df["age"].fillna(age_med, inplace=True)
df.head(20)
  #-method 2
df["age"].fillna(df["age"].median(), inplace=True)

# Task 15: Find the sum, count, mean values of the variable survived in the breakdown of pclass and gender variables.

   # -method 1
   df.groupby(["pclass","sex"]).agg({"survived":["sum","count","mean"]})

   # -method 2
   df.pivot_table(values="survived", index=["pclass","sex"], aggfunc=["sum","count","mean"])

# Task 16: Write a function that returns 1 for those under 30 and 0 for those above or equal to 30.
Using the function you wrote, create a variable named age_flag in the titanic data set. (use apply and lambda structures)

   # -method 1
    def age_30(age):
      if age<30:
          return 1
      else:
          return 0

  df["age_flag"] = df["age"].apply(lambda x: age_30(x))
  df[["age","age_flag"]].head(20)

    # -method 2
  df["age_flag"] = df["age"].apply(lambda x:1 if x<30 else 0)

# Task 17: Define the Tips dataset within the Seaborn library.

   df= sns.load_dataset("tips")
   df.head()
    df.shape

# Task 18: Find the sum, min, max and mean values of total_bill according to the categories (Dinner, Lunch) of the Time variable.

   df.groupby("time").agg({"total_bill":["sum", "min", "max", "mean"]})


# Task 19: Find the sum, min, max and mean values of total_bill according to day and time.

   df.groupby(["day", "time"]).agg({"total_bill":["sum", "min", "max", "mean"]})

# Task 20: Find the sum, min, max and mean values of total_bill and type values of lunch time and female customers according to day.

   df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum", "mean" ,"max","min"],
                                                                             "type":["sum","mean","max","min"]})

# Task 21: What is the average of orders with size less than 3 and total_bill greater than 10? (use loc)

   df.loc[(df["size"]<3) & (df["total_bill"] > 10), "total_bill"].mean() #17.18

# Task 22: Create a new variable named total_bill_tip_sum. But let this variable give the total bill and tip paid by each customer.

   df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
   df[["total_bill_tip_sum","total_bill","tip"]].head()

# Task 23: Create a total_bill_flag variable that is the average of the total_bill variable separately for men and women.
#-For women, the average of the Female scores and for men, the average of the Male scores will be taken into account. Start by writing a function that takes gender and total_bill as parameters. (Includes if-else conditions)

  f_avg=df[df["sex"]== "Female"]["total_bill"].mean() #18.05
  m_avg=df[df["sex"]== "Male"]["total_bill"].mean() #20.74

  def func(sex, total_bill):
      if sex == "Female":
          if total_bill >= f_avg :
             return 1
          else:
             return 0
      else:
          if total_bill >= m_avg:
              return 1
          else:
              return 0

  df["total_bill_flag"] = df.apply(lambda x: func(x["sex"],x["total_bill"]), axis=1)
  df.head()
# Task 24: Observe the number of people below and above the average according to gender, using the total_bill_flag variable.

  df.groupby(["sex","total_bill_flag"]).agg({"total_bill_flag":"count"})

# Task 25: Sort the data from largest to smallest according to the total_bill_tip_sum variable and assign the first 30 people to a new dataframe.

   temp_df= df.sort_values("total_bill_tip_sum", ascending=False)[:30]
   temp_df.head()