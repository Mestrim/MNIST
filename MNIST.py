import csv
import random
import math
import sys
from PIL import Image
import os
from mean_function import *
from sigmoid_function import *
from correct_percent_function import *


# IMPORTANT! To start this program you need to download datasets in CSV: mnist_train.csv and mnist_test.csv and paste them to main folder


# Testing on 28x28 px files in .bmp format
# For example, create an image in MS Paint with dimensions 28x28 px, draw a digit, and save it as a bitmap (.bmp) in a folder corresponding to that digit.
# The filename is not important, but it's crucial that, for instance, a drawn 4 is located in a folder named 4.
# Epochs
# The number of times the algorithm will learn from the same dataset." 
epochs = 3

# Choose a sensible initial learning rate. It's usually advisable to start with small values, such as 0.1, 0.01, or even 0.001, and then adjust it during training. 
learning_rate = 0.01

# It allows the activation function to be shifted up or down
bias = 0.5

# Note! The value of the variable 'learning_rows' corresponds to the number of rows in the CSV file "mnist_train.csv" (max 60000) on which the program will train.
# Setting a high value may significantly extend the program's runtime and give the impression that it has hung!!
# I suggest starting with a value of 1500 and gradually increasing it after each program run, for example, by 1000, and observing the performance.
learing_rows = 1500

# Note! The value of the variable 'testing_row' corresponds to the rows in the CSV test file "mnist_test.csv" (max 10000).
# You should start with a small amount, for example, 100, and gradually increase it, for instance, by 500.
# Setting this value too high may give the impression that the program is not working or has hung.
testing_row = 100


# Obtaining the path to the folder where the program and files are located.
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

# opening the csv training file and loading it into memory
training_data_file = open(os.path.join(__location__, 'mnist_train.csv'))
training_data_list = training_data_file.readlines()
training_data_file.close()

# opening the test file and loading it into memory
testing_data_file = open(os.path.join(__location__, 'mnist_test.csv'))
testing_data_list = testing_data_file.readlines()
testing_data_file.close()

# cells with weights (10 cells and 784 weights each)
cells = {}

# initializing random weights
m = 0
while m < 10:
    w_weights = []
    for i in range(0,784):
        n = round(random.uniform(-0.5,0.5),4)
        w_weights.append(n)
    cells[m] = w_weights
    m += 1

# uncomment below to see the weights 
#print(w_weights)

# Target 
targets = [0,1,2,3,4,5,6,7,8,9]

print()
print("START TRANING PHASE")
print("------------------------")
print("NET TRAINING INITIALIZED")
print("------------------------")

# NETWORK TRAINING 
for epoch in range(epochs):
# search the entire training list
    correct = 0
    wrong = 0
    i = 0

    while i < learing_rows:
        # searching for a goal (value in 1st column)
        label = training_data_list[i][0]
        label = int(label)

        # vector target 
        t = 0
        while t < len(targets):
            targets[t] = 0
            t += 1
        targets[label] = 1
        target = targets[label]

        # selecting the entire row including the label column
        record = training_data_list[i]
        # separating elements with a comma
        all_values = record.split(',')
        # getting rid of the newline character from the last element
        all_values[-1] = all_values[-1].strip()
        # get rid of the first element, which indicates what number in the image
        all_values.pop(0)
        input = []
        for value in all_values:
            input.append(int(value))
        j = 0
        while j < len(input):
            if input[j] > 0: 
                input[j] = 1
                j += 1
            else:
                input[j] = 0
                j += 1

        # output vector
        output = [0,0,0,0,0,0,0,0,0,0]  
        for p in cells:    
            weighted_sum = 0
            for x,w in zip(input, cells[p]):
                weighted_sum += x * w
            # adding bias to the result
            weighted_sum += bias
             # normalization of the result
            weighted_sum = sigmoid(weighted_sum)
            weighted_sum = round((weighted_sum),4)
            output[p] = weighted_sum
       
        max_value = max(output)
        index = output.index(max_value)
        
        if index == label:
            correct += 1
        else:
            wrong += 1    
            
        # weight update
        k = 0
        for key, value in cells.items():
            if key == k:
                for v in range(len(value)):
                    # new weight = old weight + lr *(target - perceptron output) * input(pixel)
                    new_weight = value[v] + learning_rate * (targets[k] - output[k]) * input[v]
                    value[v] = new_weight
                k += 1
            else:
                break
        
        # bias update
        for b in range(len(output)): 
            bias += learning_rate * (targets[b] - output[b])
            
        # Calculation of error (performance)
        meanerror = mean(output,targets)
        meanerror = round(meanerror,6)
        i += 1

    percent_correct = calculate_percent_correct(correct, wrong)

    print('START OF EPOCH')
    print(f"{epoch + 1}/{epochs}")
    print("------------------------")
    print('Loss:',meanerror)
    print('Bias',bias)
    print("correct: " ,correct)
    print("wrong: ",wrong)
    print(f"percent_correct:{round((percent_correct),2)}%")
    print("------------------------")
    print('END OF EPOCH')
    print(f"{epoch + 1}/{epochs}")
    print("------------------------")

print("NET TRAINING COMPLETED")
print("------------------------")
print("END OF TRANING PHASE")
print("------------------------")
print()
print("------------------------")
print("START TESTING PHASE")
print("------------------------")
print("NET TESTING INITIALIZED")
print("------------------------")

# NETWORK TESTING
i = 0
correct = 0
wrong = 0

while i < testing_row:
    label = testing_data_list[i][0]
    label = int(label)

    t = 0
    while t < len(targets):
        targets[t] = 0
        t += 1
    targets[label] = 1
    target = targets[label]

    record = testing_data_list[i]
    all_values = record.split(',')
    all_values[-1] = all_values[-1].strip()
    all_values.pop(0)
    input = []
    for value in all_values:
        input.append(int(value))
    j = 0
    while j < len(input):
        if input[j] > 0: 
            input[j] = 1
            j += 1
        else:
            input[j] = 0
            j += 1
    output = [0,0,0,0,0,0,0,0,0,0]  
    for p in cells:    
        weighted_sum = 0
        for x,w in zip(input, cells[p]):
            weighted_sum += x * w

        weighted_sum += bias
        weighted_sum = sigmoid(weighted_sum)
        weighted_sum = round((weighted_sum),4)
        output[p] = weighted_sum
    max_value = max(output)
    index = output.index(max_value)
    
    if index == label:
        correct += 1
    else:
        wrong += 1

    meanerror = mean(output,targets)
    meanerror = round(meanerror,6)
    i += 1

# Calculate the percentage of correct results from the test set
percent_correct = calculate_percent_correct(correct, wrong)

print()
print('TEST')
print('Loss:',meanerror)
print('Bias:',bias)
print("correct: " ,correct)
print("wrong: ",wrong)
print("percent_correct: %", round((percent_correct),2))
print()
print("------------------------")
print("NET TESTING COMPLETED")
print("------------------------")
print("END OF TESTING PHASE")
print("------------------------")
print("PICTURE TESTING INITIATED")
print("------------------------")


# Testing on .bmp files
# dictionary containing the folder name as key and files as values
file_dict = {}

for folder_number in range(10):
    folder_path = os.path.join(__location__, str(folder_number))
     # Check if the folder exists
    if os.path.exists(folder_path):
        # list of files in individual folders
        tests_file_lists = []

         # Iterate through the files in the folder   
        for test_file_name in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, test_file_name)):
                # Add the file name without extension to the list
                test_file_name_split = os.path.splitext(test_file_name)[0]
                tests_file_lists.append(test_file_name_split)
        
        file_dict[str(folder_number)] = tests_file_lists

print("Files dictionary:", file_dict)
print("------------------------")
# Answers AI
correct = 0
wrong = 0

# Open each file in the list using Image.open
for directory, file_list in file_dict.items():
    for number_test_file in file_list:
        # adding a path to the variable
        location_file = os.path.join(__location__, directory, f'{number_test_file}.bmp')
        # opening the test file
        img = Image.open(location_file)

        # add target
        correct_target = int(directory)

        final = []
        for y in range(28):
            for x in range(28):
                if img.getpixel((x,y)) == 0:
                    final.append(1)
                else:
                    final.append(0)

        # output vector
        # the values ​​in the vector correspond to the digits 0 - 9
        # individual values ​​indicate the probability of a given digit appearing in the image
        # the highest value is the neural network's indication of the most likely result
        output = [0,0,0,0,0,0,0,0,0,0]  
        for p in cells:    
            weighted_sum = 0
            for x,w in zip(final, cells[p]):
                weighted_sum += x * w
            weighted_sum += bias
            weighted_sum = sigmoid(weighted_sum)
            weighted_sum = round((weighted_sum),4)
            output[p] = weighted_sum

        max_value = max(output)
        index = output.index(max_value)

        # Target 
        if index == correct_target:
            correct += 1
        else:
            wrong += 1
        meanerror = mean(output,targets)
        meanerror = round(meanerror,6)

        print(f"Number in picture: {correct_target}")
        print('Loss:',meanerror)
        print('Bias:',bias)
        print("Correct: ", correct)
        print("Wrong: ", wrong)
        print(f"Output vector [0-9]")
        print(output)
        print(f"Highest value in vector: {max_value}")
        print(f"AI ANSWER: {index}")
        print("------------------------")
        print()

percent_correct_pictures = calculate_percent_correct(correct, wrong)

print(f"Percent correct testing data: {percent_correct}%")
print(f"Percent correct pictures: {percent_correct_pictures}%")

# uncomment below to see the weights
#print(w_weights)
    





