import os
import shutil 
from pathlib import Path
import random

random.seed(42)

base_path = "Main/clean_data"   # current folder where your CSV files are
data_folder = Path("Main/clean_data")  # <-- correct relative path

folders = ["Training", "Test", "Validation"]


for folder in folders:
    os.makedirs(os.path.join(base_path, folder), exist_ok=True)



# Lists for each group
files_45 = []
files_60 = []
files_var = []

# Loop over all CSV files
for file in data_folder.iterdir():
    name = file.name
    if "45" in name:
        files_45.append(name)
    elif "60" in name:
        files_60.append(name)
    elif "scenario" in name:
        files_var.append(name)

# Check results
"""
print("45% files:", files_45)
print("60% files:", files_60)
print("Variable scenario files:", files_var)
"""



number_of_45_files = len(files_45)
number_of_60_files = len(files_60)
number_of_var_files = len(files_var)


found = False

test_arr = []
train_arr = []
validate_arr = []


while found == False: #finds two random numbers and makes sure they aren't equal.
    n1 = random.randrange(number_of_45_files)
    n2 = random.randrange(number_of_45_files)
    if n1 != n2:
        found = True

found = False #set false again for next numbers to be selected

if n1>n2: #switches the variable names in the case that n1 is greater than n2. (Important for the pop, check below)
    placeholder = n1
    n1 = n2
    n2 = placeholder

#selects n1 for the testing files and n2 for validation. the random numbers are used as positions.
test_45 = files_45[n1]
validation_45 = files_45[n2]

#the remaining files in the list will be the training, so the previous positions assigned to testing, validation are now popped out of hte list
#n2 is guaranteed to be greater than n1 due to the if check earlier. therefore popping it first will not change
#n1 position in the list. Now the three list sets are coompleted for 45% loading condition. And same is repeated for 60% and variable loading.
files_45.pop(n2)
files_45.pop(n1)
training_45 = files_45


while found == False:
    n1 = random.randrange(number_of_60_files)
    n2 = random.randrange(number_of_60_files)
    if n1 != n2:
        found = True

found = False
    
test_60 = files_60[n1]
validation_60 = files_60[n2]

files_60.pop(n2)
files_60.pop(n1)
training_60 = files_60



while found == False:
    n1 = random.randrange(number_of_var_files)
    n2 = random.randrange(number_of_var_files)
    if n1 != n2:
        found = True
 


test_var = files_var[n1]
validation_var = files_var[n2]



files_var.pop(n2)
files_var.pop(n1)
training_var = files_var


print("Here we print the sets for each to see whether all works accordingly.")

#the [] was needed because these lists contained one single word so it was treated as a string instead of an entry
test_set = [test_45] + [test_60] + [test_var]
validation_set = [validation_45] + [validation_60] +[validation_var] 
training_set = training_45 + training_60 + training_var


print("Full Test Set:", test_set)
print("Full Validation Set:", validation_set)
print("Full Training Set:", training_set)


sets = {
    "Test":       test_set,
    "Validation": validation_set,
    "Training":   training_set,
}

#This below just sorts the files from the clean_data folder into the created folders, according to the lists developed above.
for folder_name, file_list in sets.items():
    destination = Path(base_path) / folder_name
    for filename in file_list:
        source = data_folder / filename
        shutil.move(source, destination / filename)
        print(f"Moved '{filename}' → {destination}")

print("\nAll files distributed successfully.")