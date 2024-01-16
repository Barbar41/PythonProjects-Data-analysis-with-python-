#######################
# FUNCTIONS, CONDITIONS, LOOPS, COMPREHENSIONS
#######################
# - Functions
# - Conditions
# - Loops
# - Comprehesions

#######################
# FUNCTIONS
#######################

#############################
# Function Literacy
##########################

print("a", "b")

print("a", "b", sep="__")


#############################
# Function Definition
##########################

def calculate(x):
     print(x * 2)


calculate(5)

# Let's define a function with two arguments/parameters

def summer(arg1, arg2):
     print(arg1+arg2)

summer(7,8)

summer(8,7)

summer(arg2=8,arg1=7)

####################
#docstring
####################

def summer(arg1, arg2):
     print(arg1 + arg2)


def summer(arg1, arg2):
    """

    Args:
        arg1: int, float

        arg2:int, float

    Returns:
         int,float
    """
     print(arg1 + arg2)

summer(1,3)
#######################
# Statement/Body Section of Functions
#######################

# def function_name(parameters/arguments):
# statements(function body)

def say_hi():
     print("Hello")
     print("Hi")
     print("Hello")

say_hi()

def say_hi(string):
     print(string)
     print("Hi")
     print("Hello")

say_hi("miuul")

def multiplication(a, b):
     c = a* b
     print(c)

multiplication(10,9)

# Function to count the entered values into a list

list_store =[]

def add_element(a,b):
     c = a * b
     list_store.append(c)
     print(list_store)

add_element(1,8)
add_element(18,8)
list_store

add_element(180,10)

#######################
# Default Parameters/Arguments
#######################

def divide(a,b):
     print(a/b)

divide(1,2)

def divide(a, b=1):
     print(a/b)

divide(1)

divide(10)

def say_hi(string="Hello"):
     print(string)
     print("Hi")
     print("Hello")

say_hi()
say_hi("mrb")

####################
# When Do We Need to Write Functions?
####################

# varm, moisture, charge

(56 + 15) /80
(17 + 45) /70
(52 + 45) / 80

# DRY

def calculate(varm, moisture, charge):
     print((varm + moisture)/ charge)

     calculate(98,12,78)
#######################
# Return: Using Function Outputs as Input
#######################

def calculate(varm, moisture, charge):
     print((varm + moisture)/ charge)

calculate(98,12,78) *10
type(calculate(98,12,78))

def calculate(varm, moisture, charge):
         return(varm + moisture) / charge
calculate(98, 12, 78) * 10

a= calculate(98,12,78)

def calculate(varm, moisture, charge):
     varm = varm * 2
     moisture = moisture * 2
     charge = charge * 2
     output=(varm+ moisture)/charge

     return varm, moisture, charge, output

calculate(98, 12, 78)
type(calculate(98, 12, 78))

varm, moisture, charge, output= calculate(98, 12, 78)

#######################
#- Calling a Function from Within a Function
#######################

def calculate(varm, moisture, charge):
     return int((varm + moisture) / charge)

calculate(98, 12, 78) * 10

def standardization(a,p):
     return a * 10 /100 * p * p

standardization(45,1)

def all_calculation(varm, moisture, charge, p):
     a= calculate(varm, moisture, charge)
     b= standardization(a, p)
     print(b*10)

all_calculation(1,3,5,12)

# other scenario

def all_calculation(varm, moisture, charge, a, p):
     print(calculate(varm, moisture, charge))
     b = standardization(a, p)
     print(b * 10)


all_calculation(1, 3, 5, 19,12)

#######################
# Local & Global Variables
#######################

list_store=[1,2]
type(list_store)

def add_element(a,b):
    c= a * b
    list_store.append(c)
    print(list_store)

add_element(1,9)

#######################
# CONDITIONS
####################

# True-False reminder

1 == 1
1 == 2

# if

if 1 == 1 :
     print("something")

if 1 == 2 :
     print("something")

number=11

if number == 10:
     print("number is 10")

number=10

def number_check(number):
     if number == 10:
         print("number is 10")

number_check(10)
####################
#else
####################

def number_check(number):
     if number == 10:
         print("number is 10")
     else:
         print("number is not 10")

         number_check(12)
####################
#- elif
#################

def number_check(number):
     if number > 10:
         print("greater than 10")
     elif number < 10:
         print("less than 10")
     else:
         print("equal to 10")
   number_check(10)
         number_check(6)
number_check(12)
#######################
# LOOPS
#######################
# for loop

students=["John", "Mark", "Vanessa", "Mariam"]

students[0]

students[1]

students[2]


for students in students:
     print(student)

for students in students:
     print(student.upper())

salaries=[1000,2000,3000,4000,5000]

for salaries in salaries:
print(salary)

for salaries in salaries:
print(int(salary*20/100+salary))

for salaries in salaries:
print(int(salary*30/100+salary))

for salaries in salaries:
print(int(salary*50/100+salary))

def new_salary(salary, rate):
     return int(salary*rate/100+salary)

new_salary(1500,10)
new_salary(2000,20)

for salaries in salaries:
     print(new_salary(salary, 10))

salaries2=[10700,25000,30500,40300,50200]

for salary in salaries2:
     print(new_salary(salary, 15))

salaries=[1000,2000,3000,4000,5000]

for salaries in salaries:
     if salary >=3000:
       print(new_salary(salary, 10))
     else:
       print(new_salary(salary, 20))


#Purpose We want to write a function that changes strings as follows.

# before: "hi my name is john and i am learning python"
# after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

range(len("miuul"))
range(0,5)

for i in range(0,5):
     print(i)

for i in range(len("miuul")):
     print(i)

4% 2 == 0

x="Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

def alternating(string):
     new_string=""
     #go through the indexes of the entered string
     for string_index in range(len(string)):
         If #index is even, convert it to uppercase.
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        #If index is odd, convert to lowercase
         else:
            new_string += string[string_index].lower()
      print(new_string)

alternating("miuul")
alternating(x)

#######################
# break & continue & while
####################

salaries = [1000, 2000, 3000, 4000, 5000]

for salaries in salaries:
     if salary == 3000:
         break
     print(salary)

for salaries in salaries:
     if salary == 3000:
         continue
     print(salary)

#while

number = 1
while number<5:
     print(number)
     number +=1

#######################
# Enumerate: for loop with automatic Counter/Indexer
#######################

  students=["john","mark","vanessa","mariam"]

  for students in students:
      print(student)


  for index,student in enumerate(students, 1):
     print(index,student)


A=[]
B=[]

for index,student in enumerate(students):
     if index %2 == 0:
         A.append(student)
     else:
         B.append(student)
   A.
   B.

#######################
# Application- Interview Question
#######################
# write the divide_students function
# put the students in the double index into a list
# put the students in one index into another list
# but let these two lists return as a single list.

students=["John", "Mark", "Venessa", "Mariam"]
len(students)

def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

divide_students(students)
st =divide_students(students)
st[0]

st[1]

#######################
# Writing the alternating function with enumerate
#######################

def alternating_with_enumerate(string):
     new_string = ""
     for i, letter in enumerate(string):
         if i %2 == 0:
            new_string += letter.upper()
         else:
            new_string += letter.lower()
     print(new_string)

alternating_with_enumerate("hi my name is john and i am learning python")

####################
#zip
####################


students=["John", "Mark", "Venessa", "Mariam"]
departments=["mathematics", "statics", "physics", "astronomy"]
ages=[23, 30, 26, 22]

list(zip(students, departments, ages))

#######################
# Lambda, map, filter, reduce
#######################

def summer(a, b):
     return a + b

summer(1, 3)* 9

new_sum= lambda a, b: a + b

new_sum(4,5)

# map

salaries=[1000, 2000, 3000, 4000, 5000]

def new_salary(x):
     return x * 20 / 100 + x
new_salary(1000)
new_salary(5000)

for salaries in salaries:
     print(new_salary(salary))

list(map(new_salary,salaries))

#-del new_sum
#lambda map relationship

list(map(lambda x: x*20/100+x,salaries))

list(map(lambda x: x** 2,salaries))

#Filter

list_store=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(list(filter(lambda x: x % 2 == 0, list_store))

#Reduce

from functools import reduce
list_store=[1, 2, 3, 4,]
reduce(lambda a, b: a+b, list_store)

####################### ####
# COMPREHENSIONS
####################### ###

##########################
#- List Comprehension
##########################

#---old method

salaries=[1000, 2000, 3000, 4000, 5000]

def new_salary(x):
     return x * 20 / 100 + x

for salaries in salaries:
     print(new_salary(salary))

null_list=[]
for salaries in salaries:
     null_list.append(new_salary(salary))

for salaries in salaries:
     if salary > 3000:
         null_list.append(new_salary(salary))
     else:
         null_list.append(new_salary(salary*2))

null_list

### -- with comprehensions structure

#-scenerio 1
[new_salary(salary*2) if salary <3000 else new_salary(salary) for salary in salaries]

[salary * 2 for salary in salaries]

[salary * 2 for salary in salaries if salary < 3000]

[salary * 2 if salary < 3000 else salary * 0 for salary in salaries ]

[new_salary(salary * 2) if salary < 3000 else salary * 0 for salary in salaries ]

[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries ]

#-scenerio 2

students=["John", "Mark", "Venessa", "Mariam"]

students_no=["John", "Venessa"]

[student.lower() if student in students_no else student.upper() for student in students]

[student.upper() if student not in students_no else student.lower() for student in students]

####################
# Dict Comprehensions
####################

dictionary= {"a":1,
              "b":2,
              "c":3,
              "d":4}
dictionary.keys()
dictionary.values()
dictionary.items()

{k: v **2 for (k, v) in dictionary.items()}

{k.upper(): v **2 for (k, v) in dictionary.items()}

{k.upper(): v for (k, v) in dictionary.items()}

{k.upper(): v *2 for (k, v) in dictionary.items()}

####################### ###########
# Application - Interview Question
####################### #######

# Purpose: squaring even numbers and adding them to a dictionary
# Keys will be original values and values will be modified values
#--with loop

numbers= range(10)
new_dict = {}
for n in numbers:
     if n % 2 ==0:
        new_dict[n]= n**2

{n: n ** 2 for n in numbers if n % 2 == 0}

#######################
# List & Dict Comprehension Applications
#######################

#######################
# Changing Variable Names in a Data Set
#######################

import seaborn as sns
df= sns.load_dataset("car_crashes")
df.columns

for col in df.columns:
     print(col.upper())

A = []

for col in df.columns:
     A.append(col.upper())

df.columns=A

df= sns.load_dataset("car_crashes")
  df.columns =[col.upper() for col in df.columns]

####################
# We want to add NO_FLAG to the beginning of the variables that have "INS" in their names and the others as flags.
#######################

  [col for col in df.columns if "INS" in col]

  ["FLAG_"+ col for col in df.columns if "INS" in col]

  ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns ]

df.columns=["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns ]

#######################
# The purpose is to create a dictionary with a string for key and a list for value as follows.
# We want to perform the operation only for numeric variables.
#######################

import seaborn as sns
df= sns.load_dataset("car_crashes")
df.columns

num_cols=[col for col in df.columns if df[col].dtype !="O"]

soz={}
agg_list=["mean", "min", "max", "sum"]

for col in num_cols:
     soz[col]= agg_list

# shortcut
  new_dict={col: agg_list for col in num_cols}
  df.head()
  df[num_cols].head()

  df[num_cols].agg(new_dict)