import matplotlib.pyplot as plt
#basics, line graph
x = [1,2,3,4,5,6,7]
y = [1,4,9,16,9,4,1]

x2 = [1,2,3,4,5,6,7]
y2 = [1,8,27,64,27,8,1]

plt.plot(x,y, label ='square', linewidth = 5)
plt.plot(x2, y2, label ='cube')
plt.xlabel('Plot Number')
plt.ylabel('Important Var')
plt.title('Intresting Graph\nCheck it out')
plt.legend()
plt.show()


#barcharts

x3 = [2,4,6,8,10]
y3 = [6,7,8,2,4]

x4 = [1,3,5,7,9]
y4 = [7,8,2,4,2]
plt.bar(x3, y3, label = 'bars1', color = 'r')
plt.bar(x4, y4, label = 'bars2', color = 'c')
plt.xlabel('x')
plt.ylabel('y')
plt.title('title')
plt.legend()
plt.show()

#histogram

population_ages = [22,55,62,45,21,22,34,42,32,4,99,102,54,76,36,14,65,109,16,18,57,89,75,46,65,87]
#ids = [x for x in range(len(population_ages))]
bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]

plt.hist(population_ages, bins, histtype='bar' , color = 'g', rwidth = 0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('title')
plt.legend()
plt.show()

#scatter plot

plt.scatter(x,y, label = 'scatter', color = 'k', marker = 'x', s = 500)

plt.xlabel('x')
plt.ylabel('y')
plt.title('title')
plt.legend()
plt.show()


#stackplot
days=[1,2,3,4,5]

sleeping = [7,8,6,11,7]
eating = [2,3,4,3,2]
working = [7,8,7,2,2]
playing = [8,5,7,8,13]
#we do this to create labels as we cant have labels in stack plot
plt.plot([],[],color='m', label='Sleeping', linewidth=5)
plt.plot([],[],color='c', label='Eating', linewidth=5)
plt.plot([],[],color='r', label='Working', linewidth=5)
plt.plot([],[],color='k', label='Playing', linewidth=5)

plt.stackplot(days, sleeping, eating, working, playing, colors = ['m','c','r','k'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('title')
plt.legend()
plt.show()

#pie charts
sleeping = [7,8,6,11,7]
eating = [2,3,4,3,2]
working = [7,8,7,2,2]
playing = [8,5,7,8,13]

slices = [7,2,2,13]
activities = ['sleeping','eating','working','playing']
cols = ['c','m','r','b']

plt.pie(slices, labels = activities, colors = cols, startangle = 90, shadow = True, explode= (0,0.1,0,0), autopct = '%1.1f%%')

##plt.xlabel('x')
##plt.ylabel('y')
plt.title('title')
##plt.legend()
plt.show()
