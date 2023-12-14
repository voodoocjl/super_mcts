import random
import pickle


# set random seed
random.seed(42)

# size of search space
N = 50000

search_space = []
i = 0
while i < N:
    """
    Generate a random architecture:
    r - num of layer repetitions (dim=1, n_val=2)
    q - single-qubit parametric gates (dim=7, n_val=3)
    c - categories of entangled gates (dim=7, n_val=2)
    p - positions of entangled gates (dim=7, n_val=6)
    """
    # r = [random.randint(0, 1)]
    q = [random.randint(0, 6) for _ in range(6)]
    # # c = [random.randint(0, 1) for _ in range(6)]
    # p = [random.choice([1, 2, 3, 4, 5]),
    #      random.choice([0, 2, 3, 4, 5]),
    #      random.choice([0, 1, 3, 4, 5]),
    #      random.choice([0, 1, 2, 4, 5]),
    #      random.choice([0, 1, 2, 3, 5]),
    #      random.choice([0, 1, 2, 3, 4])]
    arch = q

    if arch in search_space:
        continue

    search_space.append(arch)
    i += 1
    if i % 10000 == 0:
        print("Collected {} architectures".format(i))

with open('search_space', 'wb') as file:
    pickle.dump(search_space, file)
