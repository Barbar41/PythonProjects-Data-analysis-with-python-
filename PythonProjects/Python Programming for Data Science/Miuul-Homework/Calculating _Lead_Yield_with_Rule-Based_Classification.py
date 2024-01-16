####################### ##############
# Customer Return Calculation with Rule Based Classification
####################### ##############

# Task 1: Answer the following questions:

# Question 1: Read the persona.csv file and show general information about the data set.

import pandas as pd
pd.set_option('display.max_rows', None)
df=pd.read_csv("Python Programming for Data Science/datasets/persona.csv")
df.head()
df.info()
df.shape

# Question 2: How many unique SOURCE are there? What are their frequencies?

df["SOURCE"].nunique()
df["SOURCE"].value_counts()

# Question 3: How many unique PRICEs are there?

df["PRICE"].nunique()

# Question 4: How many sales were made from which PRICE?

df["PRICE"].value_counts()

# Question 5: How many sales were made from which country?

df["COUNTRY"].value_counts()
df.groupby("COUNTRY")["PRICE"].count()

# Question 6: How much was earned from sales in total by country?

df.groupby("COUNTRY")["PRICE"].sum() #method 1
df.groupby("COUNTRY").agg({"PRICE":"sum"}) #method 2

# Question 7: What are the sales numbers according to SOURCE types?

df["SOURCE"].value_counts()

# Question 8: What are the PRICE averages by country?

df.groupby(by=['COUNTRY']).agg({"PRICE":"mean"})

# Question 9: What are the PRICE averages according to SOURCEs?

df.groupby(by=['SOURCE']).agg({"PRICE":"mean"})

# Question 10: What are the PRICE averages in the COUNTRY-SOURCE breakdown?

df.groupby(by=['COUNTRY', 'SOURCE']).agg({"PRICE":"mean"})

# Task 2: What are the average earnings in COUNTRY, SOURCE, SEX, AGE breakdown

df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE":"mean"}).head()

# Task 3: Sort the output by PRICE
# Question 1: To better see the output in the previous question, apply the sort_values method in decreasing order according to PRICE.
# Question 2: Save the output as agg_df.

agg_df= df.groupby(by=["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).sort_values("PRICE", ascending=False)
agg_df.head()

# Task 4: Convert the names in the index to variable names
# Question 1: All variables except PRICE in the output of the third question are index names. Convert these names to variable names

agg_df= agg_df.reset_index()
agg_df.head()
agg_df.columns

# Task 5: Convert the age variable to a categorical variable and add it to agg_df

# Question 1: Convert the numeric variable Age into a categorical variable.
# Question 2: Create the ranges convincingly. For example: '0_18', '19_23', '24_30', '31_40', '41_70'

agg_df["AGE"].max()
  Let's specify where the variable #---AGE will be divided:

  bins=[0, 18, 23, 30, 40, agg_df["AGE"].max()]

  #---Let's express what the naming will be for the divided points:

  mylabels=['0-18', '19_23', '24_30', '31_40', '41_'+ str(agg_df["AGE"].max())]

  Let's divide #---AGE
  agg_df["age_cat"]= pd.cut(agg_df["AGE"], bins, labels=mylabels)
  agg_df.head()


  # Task 6: Define new level-based customers (personas).
# Question 1: Define new level-based customers (personas) and add them to the data set as a variable.
# Question 2: Name of the new variable to be added: customers_level_based
# Question 3: You need to create the customers_level_based variable by combining the observations in the output you will obtain in the previous question.

  #--Variable names:
  agg_df.columns

  #--How do we access observation values?

  for row in agg_df.values:
      print(row)

  #--putting the values of COUNTRY, SOURCE, SEX and age_cat variables side by side and concatenating them with "_"
  #--We can do this with list comprehension.
  #--Let's perform the operation to select the observation values in the loop above that we need.

[row[0].upper() + "_" + row[1].upper()+ "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]

  #--Let's add it to the data set:

  agg_df["customers_level_based"]=[row[0].upper() + "_" + row[1].upper()+ "_" + row[2].upper() + "_" + row[5 ].upper() for row in agg_df.values]
  agg_df.head()

  #--Let's remove unnecessary variables:

  agg_df= agg_df[["customers_level_based", "PRICE"]]
  agg_df.head()

  Let's make it look better by removing #--"_"

  for i in agg_df["customers_level_based"].values:
      print(i.split("_"))

  # We are one step closer to our goal.
  # There is a little problem here. There will be many same segments. For example: There may be many segments of USA_ANDROID_MALE_0_18. Let's check

  agg_df["customers_level_based"].value_counts()

  # For this reason, after groupby by segments, we should take the price averages and deduplicate the segments.

  agg_df= agg_df.groupby("customers_level_based").agg({"PRICE":"mean"})
  agg_df.head()
  agg_df.columns

  agg_df= agg_df.reset_index()

  agg_df["customers_level_based"].value_counts()

# Task 7: Divide new customers (personas) into segments.
# Question 1: Divide new customers (Example: USA_ANDROID_MALE_0_18) into 4 segments according to PRICE.
# Question 2: Add the segments to agg_df as variables with the name SEGMENT.
# Question 3: Describe the segments (Group by segment and get pricemean, max, sum).

agg_df["SEGMENT"]= pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
agg_df.groupby("SEGMENT").agg({"PRICE":"mean"})


# Task 8: Classify new customers and estimate how much revenue they can bring.

# Question 1: •To which segment does a 33-year-old Turkish woman using ANDROID belong and how much income is she expected to earn on average?

new_user= "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

# Question 2: •To which segment does a 35-year-old French woman using IOS belong and how much income is she expected to earn on average?

new_user= "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]