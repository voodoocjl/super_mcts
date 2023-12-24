import itertools
import pickle

def allPermutation(n):
    permutation = []
    for i in range(n):
        permutation.append(i)
    all_permutation = list(itertools.product(permutation, repeat=5))
    return all_permutation

n = 4
all_permutation = allPermutation(n)
result = []
for i in range(len(all_permutation)):
    temp = []
    for j in range(len(all_permutation[i])):        
        temp.append(all_permutation[i][j])
    result.append(temp)

for qubit in range(n):
    result.remove([qubit] + [(qubit+1)%n]*4)
    result.remove([qubit]*5)
print(len(result))

with open('search_space_mnist', 'wb') as file:
    pickle.dump(result, file)
