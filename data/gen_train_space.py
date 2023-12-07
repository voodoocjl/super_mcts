import csv
import pickle
import os

with open('search_space', 'rb') as file:
    search_space = pickle.load(file)
# random.shuffle(search_space)

training_data = search_space[30000:]

with open('data/train_space_2', 'wb') as file:
    pickle.dump(training_data, file)

# with open('search_space_shuffle', 'wb') as file:
#     pickle.dump(search_space, file)