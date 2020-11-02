
from funct_project import functs

a=read_csv('F:/iris.csv')

#univariate plots -box and whisker plots
pyplot.figure(figsize=(3,3))
#title
pyplot.title('box plots of all attributes')
pyplot.boxplot(a)
pyplot.show()


    