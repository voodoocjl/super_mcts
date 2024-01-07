import pickle
import random

with open('search_space_mnist', 'rb') as file:
    space = pickle.load(file)

# set random seed
random.seed(42)

# size of search space
N = 20000

search_space = []
i = 0
while i < N:
    first = random.choice(space)
    second =random.choice(space)
    while second[0] == first[0]:
        second =random.choice(space)
    search_space.append([first, second])
    i += 1

with open('search_space_mnist_2steps', 'wb') as file:
    pickle.dump(search_space, file)

print(len(search_space))