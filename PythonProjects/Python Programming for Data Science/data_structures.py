#########
# DATA STRUCTURES
########
#- Introduction to Data Structures and Quick Summary
#- Numbers: int, float, complex
#- Character Arrays (Strings): str
#- Boolean(TRUE-FALSE):bool
#- List
#- Dictionaty
#- Tuple
#- Set

# Numbers: integer
x=46
type(x)

# Numbers: float
x=10.3
type(x)

# Numbers: complex
x=2j+1
type(x)

# String
x= "Hello ai Era"
type(x)

#boolean
True

False

type(True)

5==4

3==2

type(3 == 2)

# Lists

x=["btc","eth","xrp"]
type(x)

# Dictionary

x={"name":"Peter","Age":36}
type(x)

#tuple
x =("python","ml","ds")
type(x)

#Set
x= {"python","ml","ds"}
type(x)

# List, tuple, set and dictionary structures are also called Python collections (Arrays).

#######################
# Numbers: int, float, complex
#######################

a=5
b=10.5

a*3
a/7
a*b/10
a**2

###########
# Change types
##########

int(b)
float(a)

int(a*b/10)

c=a*b/10

c

int(c)
#############################
# Strings
##########################

print("John")
"John"

name="John"
name

#############################
# Multi-Line Strings
##########################

long_str=""" Data Structures:Quick Summary,
Numbers: int, float, complex,
Strings:str,
List, Dictionary, Tuple, Set,
Boolean(True-False):bool"""

long_str

#############################
# Accessing Elements of Strings
##########################

name
name[0]
name[3]

#############################
# Slice Operation on Character Strings
##########################

name[0:2]
long_str[0:10]

#############################
# Querying a Character in a String
##########################

long_str

"data" in long_str

"Data" in long_str

"bool" in long_str

####################
# String Methods
####################

dir(str)

#######################
# Len
#######################

name="John"
type(name)
type(len)

len(name)
len("barisbayar")

#############################
# Upper() & lower(): small-to-large conversions
#############################

"miuul".upper()
"miuul".lower()

# Type(upper)
# Type(upper())

#############################
# Replace: replaces characters
#############################

hi="Hello AI Era"
hi.replace("l","p")

#############################
# Split
#############################

"Hello AI Era".split()

#############################
# Strip: trims
#############################

" ofofo ".strip()
"ofofo".strip("o")

#############################
# Capitalize: capitalizes the first letter
#############################

"foo".capitalize()

dir("foo")

"foo".startswith("f")

####################
# List
#################
#-Replaceable
#-It is sequential. Index operations can be performed.
#-It is inclusive



notes=[1,2,3,4]
type(notes)

names=["a","b","v","d"]
not_nam=[1,2,3,"a","b", True,[1,2,3]]

not_nam[0]

not_nam[5]

not_nam[6][1]

type(not_nam[6])

type(not_nam[6][1])

notes[0]

notes[0]=99

notes

not_nam[0:4]

####################
# List Methods
####################

dir(notes)


####################
# Len: built-in python function, size information.
#################

len(notes)
len(not_nam)

####################
# Append: adds elements
#################

notes
notes.append(100)
notes

####################
# Pop: deletes elements according to index
#################

notes.pop(0)
notes

####################
# Insert: adds to the index
#################

notes.insert(2, 99)
notes

#############################
# Dictionary
##############

# Can be changed
# Unordered. (Sorted after 3.7)
# Container

# Key-value

dictionary = {"REG":"Regression",
               "LOG":"Logistic Regression",
               "CART":"Classification and Reg"
               }

dictionary["REG"]

dictionary={"REG":["RMSE",10],
             "LOG":["MSE",20],
             "CART":["SSE",30]}

dictionary["REG"]

dictionary["CART"][1]

####################
# Key Query
####################

"REG" in dictionary
"ANN" in dictionary

####################
# Accessing Value by Key
####################

dictionary["REG"]
dictionary.get("REG")

####################
# Changing Value
####################

dictionary["REG"]=["ANN",10]
dictionary

####################
# Access to All Keys
####################

dictionary.keys()

####################
# Accessing All Values
####################

dictionary.values()

####################
# Converting All Pairs into a Tuple List
####################

dictionary.items()

####################
# Updating Key-Value
####################

dictionary.update({"REG":11})

####################
# Adding New Key-Value
####################

dictionary.update({"RF":10})

dictionary

#######################
# Tuple
#######################
#-Cannot be changed
#-Sequential
#-It is inclusive

t= ("john","mark",1,2)
type(t)

t[0]
t[0:3]

t[0]=99

t=list(t)
t[0]=99
t=tuple(t)
t

#######################
# Set
#######################
#-Replaceable
#-Unordered+Unique.
#-It is inclusive.

#############################
# Difference(): Difference of two sets
##########################

set1=set([1,3,5])
set2=set([1,2,3])
type(set1)

# Those in set1 but not in set2
set1.difference(set2)
# Those in set2 but not in set1
set2.difference(set1)

####################
# Symmetric_difference(): Those that are not relative to each other in both sets.

set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

#################
# Intersection():Intersection of two sets
#################

set1=set([1,3,5])
set2=set([1,2,3])

set1.intersection(set2)
set2.intersection(set1)

set1& set2

#################
# Union(): Union of two sets
#################
set1.union(set2)
set2.union(set1)

#############################
# Isdisjoint(): Is the union of two clusters empty?
#############################

set1=set([7,8,9])
set2=set([5,6,7,8,9,10])

set1.isdisjoint(set2)
set2.isdisjoint(set1)

#############################
# Isissubset(): Is a set a subset of another set?
#############################

set1.issubset(set2)
set2.issubset(set1)

#############################
# Isissuperset((): Does one cluster cover another cluster?
#############################

set2.issuperset(set1)
set1.issuperset(set2)