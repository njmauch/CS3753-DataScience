import numpy as np
from math import sqrt
import pdb
from csv import reader
from random import seed
from random import randrange
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

##%% Question 1
print("Question 1")
x = np.array([0.5, 2, 3])
y = np.array([1, 2.5, 3])

model = linear_model.LinearRegression()
model = model.fit(x.reshape(-1,1), y.reshape(-1,1))

print(model.intercept_)
print(model.coef_)

slope = 1
intercept = 0.5
SSE1 = ((y - (intercept + (slope * x))).dot(y- (intercept + (slope * x))))
slope = 1
intercept = 1
SSE2 = ((y - (intercept + (slope * x))).dot(y- (intercept + (slope * x))))
slope = 0.8
intercept = 0.3
SSE3 = ((y - (intercept + (slope * x))).dot(y- (intercept + (slope * x))))
slope = 0.8
intercept = 0.7
SSE4 = ((y - (intercept + (slope * x))).dot(y- (intercept + (slope * x))))

print("Linear Regression A: {:0.2f}".format(SSE1))
print("Linear Regression B: {:0.2f}".format(SSE2))
print("Linear Regression C: {:0.2f}".format(SSE3))
print("Linear Regression D: {:0.2f}".format(SSE4))
print("The best is D")

##%% Question 2
print("\nQuestion 2")
print("2. Ridge and lasso regression are simple techniques to prevent overfitting in linear regression")
print("Ridge regression can provide better long-term predictions and does better when most variables are useful")
print("Lasso regression is less likely to overfit and better prediction performance on new data. It can also exclude useless variables from equations")



##%% Question 3
print("\n\nQuestion 3")
def gini(node):
    total = node["C1"] + node["C2"]
    return 1 - (node["C1"]/total)**2 - (node["C2"]/total)**2
def giniChildren(node1, node2, gini_n1, gini_n2):
    total_n1 = node1["C1"] + node1["C2"]
    total_n2 = node2["C1"] + node2["C2"]
    return (total_n1/(total_n1 + total_n2) * gini_n1) + (total_n2/(total_n1 + total_n2) * gini_n2)
def entropy(node):
    total = node["C1"] + node["C2"]
    return (-(node["C1"]/total) * np.log2(node["C1"]/total)) - ((node["C2"]/total) * np.log2(node["C2"]/total))
def gain(ent_parent, ent_n1, ent_n2, node1, node2):
    total_n1 = node1["C1"] + node1["C2"]
    total_n2 = node2["C1"] + node2["C2"]
    n = total_n1 + total_n2
    return ent_parent - (total_n1/n * ent_n1 + total_n2/n * ent_n2)
def error(node):
    total = node["C1"] + node["C2"]
    p_c1 = node["C1"]/total
    p_c2 = node["C2"]/total
    return 1.0 - np.max([p_c1, p_c2])
def split_error(err_parent, err_n1, err_n2, node1, node2, parent_node):
    total_parent = parent_node["C1"] + parent_node["C2"]
    total_n1 = node1["C1"] + node1["C2"]
    total_n2 = node2["C1"] + node2["C2"]
    return err_parent - (total_n1/total_parent * err_n1 + total_n2/total_parent * err_n2)

A = {"Parent": {"C1": 20, "C2": 10}, "Node N1": {"C1": 3, "C2": 9}, "Node N2": {"C1": 17, "C2": 1}}
B = {"Parent": {"C1": 20, "C2": 10}, "Node N1": {"C1": 10, "C2": 5}, "Node N2": {"C1": 10, "C2": 5}}
C = {"Parent": {"C1": 20, "C2": 10}, "Node N1": {"C1": 19, "C2": 2}, "Node N2": {"C1": 1, "C2": 8}}
print("Split A:")
gini_parent = gini(A["Parent"])
entropy_parent = entropy(A["Parent"])
error_parent = error(A["Parent"])
print("Parent Node Gini = {:0.3f}".format(gini_parent))
gini_n1 = gini(A["Node N1"])
entropy_n1 = entropy(A["Node N1"])
error_n1 = error(A["Node N1"])
gini_n2 = gini(A["Node N2"])
entropy_n2 = entropy(A["Node N2"])
error_n2 = error(A["Node N2"])
gini_children = giniChildren(A["Node N1"], A["Node N2"], gini_n1, gini_n2)
print("Gini Children = {:0.3f}".format(gini_children))
entropy_gain = gain(entropy_parent, entropy_n1, entropy_n2, A["Node N1"], A["Node N2"])
error_split_a = split_error(error_parent, error_n1, error_n2, A["Node N1"], A["Node N2"], A["Parent"])
print("Information gain = {:0.3f}".format(entropy_gain))
print("The split error = {:0.3f}".format(error_split_a))

print("Split B:")
gini_parent = gini(B["Parent"])
entropy_parent = entropy(B["Parent"])
error_parent = error(B["Parent"])
print("Parent Node Gini = {:0.3f}".format(gini_parent))
gini_n1 = gini(B["Node N1"])
entropy_n1 = entropy(B["Node N1"])
error_n1 = error(B["Node N1"])
gini_n2 = gini(B["Node N2"])
entropy_n2 = entropy(B["Node N2"])
error_n2 = error(B["Node N2"])
gini_children = giniChildren(B["Node N1"], B["Node N2"], gini_n1, gini_n2)
print("Gini Children = {:0.3f}".format(gini_children))
entropy_gain = gain(entropy_parent, entropy_n1, entropy_n2, B["Node N1"], B["Node N2"])
error_split_b = split_error(error_parent, error_n1, error_n2, B["Node N1"], B["Node N2"], B["Parent"])
print("Information gain = {:0.3f}".format(entropy_gain))
print("The split error = {:0.3f}".format(error_split_b))

print("Split C:")
gini_parent = gini(C["Parent"])
entropy_parent = entropy(C["Parent"])
error_parent = error(C["Parent"])
print("Parent Node Gini = {:0.3f}".format(gini_parent))
gini_n1 = gini(C["Node N1"])
entropy_n1 = entropy(C["Node N1"])
error_n1 = error(C["Node N1"])
gini_n2 = gini(C["Node N2"])
entropy_n2 = entropy(C["Node N2"])
error_n2 = error(C["Node N2"])
gini_children = giniChildren(C["Node N1"], C["Node N2"], gini_n1, gini_n2)
print("Gini Children = {:0.3f}".format(gini_children))
entropy_gain = gain(entropy_parent, entropy_n1, entropy_n2, C["Node N1"], C["Node N2"])
error_split_c = split_error(error_parent, error_n1, error_n2, C["Node N1"], C["Node N2"], C["Parent"])
print("Information gain = {:0.3f}".format(entropy_gain))
print("The split error = {:0.3f}".format(error_split_c))
print("The conclusion is that split C had the best results for all 3 methods, it had the lowest value for the gini index, 0.180, and the highest values for the information gain, 0.450, and split error, 0.233")

# Load a CSV file
# Load a CSV file
print("Question 4")
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
        print('Class %s ID => %d' % (value, i))
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    ### your code starts
    distance = 0.0
    for x in range(len(row1) - 1):
        distance += (row1[x] - row2[x])**2
    ### your code ends
    return sqrt(distance)


# Locate the most similar neighbors and return the list of neighbors
def get_neighbors(train, test_row, num_neighbors):
    ### your code starts
    distances = []
    for train_row in train:
        distances.append((train_row, euclidean_distance(test_row, train_row)))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for x in range(num_neighbors):
        neighbors.append(distances[x][0])
    ### your code ends
    return neighbors


# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors):
    ### your code starts
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output = [row[-1] for row in neighbors]
    prediction = max(set(output), key=output.count)
    ### your code ends
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification(train, row, num_neighbors)
        predictions.append(output)
    return (predictions)


# Test the kNN on the Iris Flowers dataset
seed(1)
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)

# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)

# evaluate algorithm
n_folds = 5
num_neighbors = 5
scores = evaluate_algorithm(dataset, k_nearest_neighbors, n_folds, num_neighbors)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

