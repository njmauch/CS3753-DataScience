import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

#%% Q1
print("Question 1")
matrixA = np.array([[2, 8, 4], [5, 4, 2]])
matrixB = np.array([[4, 1], [6, 4], [5, 3]])
matrixC = np.array([[4, 1, 2], [6, 4, 3], [5, 3, 4]])
matrixD = np.array([[4, 1, 2], [6, 4, 3]])

try:
    result1 = np.dot(matrixA, matrixB)
    print("A∙B = ", end='')
    print(result1)
except ValueError:
    print("Can't calculate A∙D")
    
try:
    result2 = np.dot(matrixA, matrixC)
    print("A∙C =", end='')
    print(result2)
except ValueError:
    print("Can't calculate A∙C")
    
try:
    result3 = np.dot(matrixA, matrixD)
    print("A∙D = ", end='')
    print(result3)
except ValueError:
    print("Can't calculate A∙D")
    
#%% Q2
print("Question 2")
x = np.array([ 50, 68, 74, 70, 65, 61, 63, 74, 62])
y = np.array([170, 173, 209, 130, 215, 127, 108, 152, 183])

def zscore(array1):
    return (array1 - np.mean(array1))/np.std(array1)

xScores = zscore(x)
yScores = zscore(y)

plt.scatter(xScores, yScores)
plt.show

coefficient = (xScores.dot(yScores))/len(xScores)
print("The Pearsons correlation coefficient = ", coefficient)
npCoefficient = np.corrcoef(x, y)[0, 1]
print("The coeffecient calculated with NumPy = ", npCoefficient)
print("The results are the exact same")

#%% Q3
print("Question 3")
x = np.array([ 50, 68, 74, 70, 65, 61, 63, 74, 62, 20])
y = np.array([170, 173, 209, 130, 215, 127, 108, 152, 183, 800])

npCoefficient = np.corrcoef(x, y)[0, 1]
print("The coeffecient calculated with NumPy = ", npCoefficient)
xRank = np.argsort(np.argsort(x))
yRank = np.argsort(np.argsort(y))
spearmanCoefficeint = np.corrcoef(xRank, yRank)
print("The Spearman Rank correlaction coeffection = ", spearmanCoefficeint[0, 1])
print("The coeffections are not equal.  Because of the outlier value of (20,800) the coeffecients are going to be off. When we have a value like that the Spearman Rank correlation with work better")
print("3d: -1 shows an inverse relationship, 0 shows no correlation, and 1 shows a positive linear relationship")

#%% Q4
print("Question 4")
value4a = norm.cdf(0.5) - norm.cdf(-0.5)
print("4a = ", value4a)
value4b = 2*(norm.cdf(0) - norm.cdf(-0.5)) or 2*(norm.cdf(0.5) - norm.cdf(0))
print("4b = ", value4b)

#%% Q5
print("Question 5")
print("5a: X' ~ ( aμ+b, (aσ)^2)")
print("5b: Z = (X- μ)/σ")
value5c = norm.cdf(7, 5, 3) - norm.cdf(2, 5, 3)
print("5c = ", value5c)
value5d = norm.cdf(1.5) - norm.cdf(-1.5)
print("5d = ", value5d)

#%% Q6
print("Question 6")
print("6a: P(A ∪ B) = P(A) + P(B) – P(A ∩ B)")
print("6b: P(A | B) = P(A ∩ B) / P(B)")
print("6c: P(A ∩ B) = P(B) * P(A | B)")
print("6d: If A and B are independent, P(A ∩ B) = P(A) * P(B)")

#%% Q7
print("Question 7")
print("P(d = even) = 1/2")
print("P(d < 5) = 1/2")
print("P(d = even) * P(d < 5) = 1/4")
print("P(d = even ∩ d < 5) = 2/8 = 1/4")
print("P(d = even) * P(d < 5) = P(d = even ∩ d < 5)")
print("Therefore, P(d = even) and P(d < 5) are independent.")

#%% Q8
print("Question 8")
print("P(loaded|6666) = P(6666|loaded)*P(loaded)/P(6666)")
print("= P(6666|loaded)*P(loaded)/(P(6666|loaded)*P(loaded) + P(6666|fair)*P(fiar))" )
print("= (0.5**4) * 0.05 / ((0.5**4) * 0.05 + ((1/6)**4)*0.95)")
print("=0.81")

#%% Q9
print("Question 9")
print("A: have the disease, P(A) = 1/10,000")
print("B: test positive")
print("P(A)+P(neg(A)) =1")
print("P(B)+P(neg(B)) =1")
print("9a: P(B | A) = 0.999; P(B | neg(A)) = 0.0002;")
print("Bayes Theorem, P(A | B) = P(B | A) * P(A) / P(B)")
print("=P(B|A)*P(A)/(P(B|A)*P(A) + P(B|neg(A))*P(neg(A)))")
print("= 33.3%")
print("9b: P(neg(A)| neg(B)) = P(neg(B)|neg(A))*P(neg(A))/P(neg(B))")
print("= (1-P(B|neg(A)))*(1-P(A))/(1-P(B))")
print("=0.99999")

#%% load data
print("Question10")
measles=pd.read_csv('Measles.csv',header=None).values
mumps=pd.read_csv('Mumps.csv',header=None).values
chickenPox=pd.read_csv('chickenPox.csv',header=None).values

# close all existing floating figures
plt.close("all")

#%% Q10.a. plot annual total measles cases in each year
plt.figure()
plt.title("Fig 9.a: NYC measles cases")
# complete this part here
plt.plot(measles[:,0], measles[:,1:].sum(1), "-*")
plt.xlabel("Year")
plt.ylabel("Number of cases")
plt.show()

#%% Q10.b bar plot average mumps cases for each month of the year

plt.figure()
plt.title('Fig 9.b: Average monthly mumps cases')
# complete this part here
plt.bar(range(1,13), mumps[:, 1:].mean(0))
plt.xlabel("Month")
plt.ylabel("Average number of cases")
plt.show()

#%% Q10.c scatter plot monthly mumps cases against measles cases
mumpsCases = mumps[:, 1:].reshape(41*12)
measlesCases = measles[:, 1:].reshape(41*12)

plt.figure()
plt.title('Fig 9.c: Monthly mumps vs measles cases')
# complete this part here
plt.scatter(mumpsCases, measlesCases)
plt.xlabel("Number of Mumps Cases")
plt.ylabel("Number of Measles Cases")
plt.show()

#%% Q10.d plot monthly mumps cases against measles cases in log scale
plt.figure()
plt.title('Fig 9.d: Monthly mumps vs measles cases (log scale)')
# complete this part
plt.loglog(mumpsCases, measlesCases, ".")
plt.xlabel("Number of Mumps cases")
plt.ylabel("Number of Measels cases")
plt.show()


#%% Answer to Q10.e

# complete this part here
answer = "When done with the linear plot the relationship can be more difficult to tell."

print('\n\nAnswer to Q9.e: ' + answer)