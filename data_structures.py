
# integer
x = 46
type(x)

# float
x = 10.3
type(x)

# complex
x = 2j + 1
type(x)

# String
x = "Hello ai era"
type(x)

# Boolean
True
False
type(True)

# List
x = ["btc", "eth", "xrp"]
type(x)

# Sözlük (dictionary)
x = {"name": "Peter", "Age": 36}
type(x)

# Tuple
x = ("python", "ml", "ds")
type(x)

# Set
x = {"python", "ml", "ds"}
type(x)


# Strings


print("John")
print('John')

"John"
name = "John"
name = 'John'

long_str = """Veri Yapıları: Hızlı Özet, 
Sayılar (Numbers): int, float, complex, 
Karakter Dizileri (Strings): str, 
List, Dictionary, Tuple, Set, 
Boolean (TRUE-FALSE): bool"""

# Slice
name[0:2]
long_str[0:10]


# len

name = "john"
type(name)
type(len)

len(name)
len("vahitkeskin")
len("miuul")


# upper() & lower()

"miuul".upper()
"MIUUL".lower()
# type(upper)
# type(upper())

# replace
hi = "Hello AI Era"
hi.replace("l", "p")

# split
"Hello AI Era".split()

# strip
" ofofo ".strip()
"ofofo".strip("o")

# capitalize
"foo".capitalize()

dir("foo")

"foo".startswith("f")


# Liste (List)


notes = [1, 2, 3, 4]
type(notes)
names = ["a", "b", "v", "d"]
not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]]

not_nam[0]
not_nam[6][1]

type(not_nam[6])

notes[0] = 99

not_nam[0:4]

# len

len(notes)
len(not_nam)

# append

notes
notes.append(100)

# pop

notes.pop(0)


# insert

notes.insert(2, 99)

# Sözlük (Dictonary)
# key-value

dictionary = {"REG": "Regression",
              "LOG": "Logistic Regression",
              "CART": "Classification and Reg"}

dictionary["REG"]


dictionary = {"REG": ["RMSE", 10],
              "LOG": ["MSE", 20],
              "CART": ["SSE", 30]}

dictionary = {"REG": 10,
              "LOG": 20,
              "CART": 30}

dictionary["CART"][1]

dictionary["REG"]
dictionary.get("REG")

dictionary["REG"] = ["YSA", 10]

dictionary.keys()
dictionary.values()
dictionary.items()

dictionary.update({"REG": 11})

dictionary.update({"RF": 10})

# Demet (Tuple)

t = ("john", "mark", 1, 2)
type(t)

t[0]
t[0:3]

t[0] = 99

t = list(t)
t[0] = 99
t = tuple(t)

# Set

# difference(): İki kümenin farkı

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])


set1.difference(set2)
set1 - set2


# symmetric_difference()

set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

# intersection()

set1 = set([1, 3, 5])
set2 = set([1, 2, 3])

set1.intersection(set2)
set2.intersection(set1)

set1 & set2


# union()

set1.union(set2)
set2.union(set1)

# isdisjoint()

set1 = set([7, 8, 9])
set2 = set([5, 6, 7, 8, 9, 10])

set1.isdisjoint(set2)
set2.isdisjoint(set1)

# isdisjoint()

set1.issubset(set2)
set2.issubset(set1)

# issuperset()

set2.issuperset(set1)
set1.issuperset(set2)








