string = 'Welcome to Python Programming'
print(string)
print(string[11:17])
print(string.replace("Programming", "Environment"))

list1 = []
list1.append(1)
list1.append(2)
list1.append(3)
list1.append(4)
list1.extend((5,6))
list1.extend(['perfect', 'wonderful'])
list2 = [[7,8],[9,10]]
list1 = list1 + list2
list1.extend([8.5, 7, 'code', 'software'])
print(list1[-5:])
del list1[3:6]

Tuple1 = (1, 2, 3, 4)
Tuple2= ('Python', 'for', 'kids')
Tuple1 = Tuple1 + Tuple2
print(Tuple1[3:])

Dict = {}
Dict[0] = 'Python'
Dict[1] = 'Programming'
Dict[2] = 'Funny'
Dict[1] = 'Very'
print(Dict.keys())
print(Dict.values())
Dict.pop(2)
print("Is key 2 in Dict? ", end="")
print(2 in Dict)
for key, value in Dict.items():
    temp = [key, value]
    print(temp, end="")