import itertools
import pickle

def allPermutation(n):
    permutation = []
    for i in range(n):
        permutation.append(i)
    all_permutation = list(itertools.product(permutation, repeat=5))
    return all_permutation

all_permutation = allPermutation(7)
result = []
for i in range(len(all_permutation)):
    temp = [5]
    for j in range(len(all_permutation[i])):
        # temp.append(str(all_permutation[i][j]))
        temp.append(all_permutation[i][j])
    result.append(temp)
result.remove([5,6,6,6,6,6])
print(len(result))

with open('5_space', 'wb') as file:
    pickle.dump(result, file)
