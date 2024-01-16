# Task 1: Examine the data structures of the given values

x = 8
type(x)

y = 3.2
type(y)

z= 8j+18
type(z)

a="Hello World"
type(a)

b=True
type(b)

c=23 < 22
type(c)

l=[1,2,3,4]
type(l)

d={"Name":"Jake",
    "Age":27,
    "Address":"Downtown"}
type(d)

t=("Machine Learning","Data Science")
type(t)

s={"Python","Machine Learning","Data Science"}
type(s)

# Task 2: Convert all letters of the given string expression to uppercase. Put space instead of commas and periods, separate them word by word.

text="The goal is to turn data into information, and information into insight"

text.upper().replace("," , " ").replace("." , " ")

text.upper().replace("," , " ").replace("." , " ").split()

# Task 3: Follow the steps below to the given list.

lst=["D","A","T","A","S","C","I","E","N","C","E"]

#Step 1: Look at the number of elements of the given list.
len(lst)

#Step 2: Call the elements at the zeroth and tenth index.
lst[0]
lst[10]

#Step 3: Create a list ["D", "A", "T", "A"] from the given list.

lst[0:4]

#Step 4: Delete the element at the eighth index.
lst.pop()
lst

#Step 5: Add a new element.
lst.append(101)
lst

#Step 6: Add the "N" element again to the eighth index.

lst.insert(8,"N")
lst

# Task 4: Apply the following steps to the given dictionary structure.

dict={'Christian':["America",18],
       'Daisy':["England",12],
       'Antonio':["Spain",22],
       'Dante':["Italy",25]}

#Step 1: Access key values.
dict.keys()

#Step 2: Access the values.
dict.values()

#Step 3: Update the value 12 of Daisy key to 13.
dict.update({"Daisy":["England",13]})
dict

#Step 4: Add a new value with the key value Ahmet value [Turkey,24].
dict["Ahmet"]=["Turkey",24]
dict

#Step 5: Delete Antonio from dictionary
dict.pop("Antonio")
dict

# Task 5: Write a function that takes a list as an argument, assigns the odd and even numbers in the list to separate lists, and returns these lists.

l=[2,13,18,93,22]

def func(list):
     cift_list=[]
     odd_list=[]

     for i in list:
         if i %2 ==0:
             cift_list.append(i)
         else:
         odd_list.append(i)
     return even_list, odd_list


cift_list, odd_list=func(l)

# Write a function that returns.


# Task 6: Using the List Comprehension structure, convert the names of the numeric variables in the car_crashes data to uppercase letters and add NUM at the beginning

import seaborn as sns
df=sns.load_dataset("car_crashes")
df.head()

df.columns

df["total"].dtype

df.info


["NUM_" + col.upper() if df[col].dtype !="O" else col.upper() for col in df.columns]


# Task 7: Using the List Comprehension structure, write "FLAG" at the end of the names of variables that do not contain "no" in their names in the car_crashes data.

[col.upper()+ "_FLAG" if "no" not in col else col.upper()for col in df.columns]

# Task 8: Using the List Comprehension structure, select the names of the variables that are DIFFERENT from the variable names given below and create a new dataframe

og_list =["abbrev","no_previous"]
new_cols = [col for col in df.columns if col not in og_list]
new_df= df[new_cols]
new_df.head()