import itertools
import pickle
import numpy as np

def allPermutation(n, numbers = None):
    permutation = []
    if numbers == None:
        for i in range(1, n+1):
            permutation.append(i)
    else:
        permutation = numbers
    all_permutation = list(itertools.product(permutation, repeat = layers+1))  # code length 4 layers is 5, 5 layers is 6...
    # convert to list
    result = []
    for i in range(len(all_permutation)):
        temp = []
        for j in range(len(all_permutation[i])):        
            temp.append(all_permutation[i][j])
        result.append(temp)
    return result

n = 4  # qubit number
# # generate sign
# layers = 3
# sign = allPermutation(n, [1, 0])
# # print(len(sign))

# # generate arch
# layers = 4
# result = allPermutation(n)
# lst = [i for i in range(1, n+1)]
# lst = np.roll(lst, -1)
# for qubit in range(1, n+1):
#     # result.remove([qubit] + [lst[qubit-1]] * layers)
#     result.remove([qubit] * (1+layers))
# # print(len(result))


# results =[]
# for item in sign:
#     tmp = [1] + item  # the first number is always positive
#     results.append(np.array(tmp) * np.array(result))
# results = np.array(results).reshape((-1, 1+layers)).tolist()


# for qubit in range(1, n+1):
#     results.remove([qubit] + [lst[qubit-1]] * layers)

# with open('search_space_4_layers_pm', 'wb') as file:
#     pickle.dump(results, file)

# print(len(results))

first = [i for i in range(1, n+1)]
others = [0, 1]
layers = 2*4

all_permutation = list(itertools.product(first, *([others]*layers)))
result_list = []
for i in range(len(all_permutation)):
    temp = []
    for j in range(len(all_permutation[i])):        
        temp.append(all_permutation[i][j])
    result_list.append(temp)

for qubit in range(1, n+1):
    result_list.remove([qubit] + [1] * layers)

with open('search_space_mnist_single', 'wb') as file:
    pickle.dump(result_list, file)
