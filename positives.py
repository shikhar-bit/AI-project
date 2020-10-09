#getting positive values ina list

list1=[1,-1,2,-2]
c=[]

for a in list1:
    if a<0:
        continue
    c.append(a)

print(list1,c)