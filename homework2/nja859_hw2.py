import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("fdata.csv")

gArray = np.array(data)

dateList = gArray[:, 0]
openList = gArray[:, 1]
highList = gArray[:, 2]
lowList = gArray[:, 3]
closeList = gArray[:, 4]
plt.plot(dateList, openList, 'b-', 
         dateList, highList, 'C1',
         dateList, lowList, 'g-',
         dateList, closeList, 'r-')

plt.xlim([dateList[0], dateList[-1]])
plt.legend(['Open', 'High', 'Low', 'Close'])
plt.show()

plt.plot(dateList, closeList, 'ro-')
plt.title("Closing stock value of Alphabet Inc.")
plt.ylabel("Closing Value")
plt.xlabel("Date")
plt.minorticks_on()
plt.grid(which='major',linestyle='-', linewidth='0.5', color='red', )
plt.grid(which='minor',linestyle=':', linewidth='0.5', color='black')
plt.show()

languageList = ['Java', 'Python', 'PHP', 'JavaScript', 'C#', 'C++']
popularity = [22.2, 17.6, 8.8, 8, 7.7, 6.7]
plt.title("Popularity of Programming Language \n Worldwide Oct 2017 compared to a year ago")
plt.ylabel("Popularity")
plt.xlabel("Languages")
plt.bar(languageList, popularity)
plt.minorticks_on()
plt.grid(which='major',linestyle='-', linewidth='0.5', color='red', )
plt.grid(which='minor',linestyle=':', linewidth='0.5', color='black')
plt.show()


plt.title("Popularity of Programming Language \n Worldwide Oct 2017 compared to a year ago")
plt.ylabel("Popularity")
plt.xlabel("Languages")
plt.barh(languageList, popularity, color='green')
plt.minorticks_on()
plt.grid(which='major',linestyle='-', linewidth='0.5', color='red', )
plt.grid(which='minor',linestyle=':', linewidth='0.5', color='black')
plt.show()

weight1 = [67,57.2,59.6,59.64,55.8,61.2,60.45,61,56.23,56]
height1 = [101.7,197.6,98.3,125.1,113.7,157.7,136,148.9,125.3,114.9]
weight2 = [61.9,64,62.1,64.2,62.3,65.4,62.4,61.4,62.5,63.6]
height2 = [152.8,155.3,135.1,125.2,151.3,135,182.2,195.9,165.1,125.1]
weight3 = [68.2,67.2,68.4,68.7,71,71.3,70.8,70,71.1,71.7]
height3 = [165.8,170.9,192.8,135.4,161.4,136.1,167.1,235.1,181.1,177.3]
plt.scatter(weight1, height1, marker='*', color='red')
plt.scatter(weight2, height2, marker='*', color='green')
plt.scatter(weight3, height3, marker='*', color='blue')
plt.title("Group Wise Weight vs Height")
plt.ylabel("Height")
plt.xlabel("Weight")
 plt.show()