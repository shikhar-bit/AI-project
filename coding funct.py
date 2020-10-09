#assigning no. in a list

a=0
c=[]
while a<10:
    c.append(a)
    a=a+1

print (c)

#accessing elements from a tuple

x=('python','java','c')

print('my tuple is '+'  '.join(x))
for a in x:
    
    print(a)
    
#deleting elements from dictionary

y={1:'python',2:'java'}

y.pop(2)
print(y)


