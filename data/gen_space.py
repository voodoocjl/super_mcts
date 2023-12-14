import pickle
import os, json

# with open('search_space_3', 'rb') as file:
#     search_space = pickle.load(file)
# # random.shuffle(search_space)
# print(len(search_space))

dir_path = os.path.dirname(os.path.realpath(__file__))
dataset_file = os.path.join(dir_path, 'mosi_dataset')

with open(dataset_file, 'rb') as file:
    dataset = pickle.load(file)
# random.shuffle(search_space)
print(len(dataset))

search_space = []
for arch, mae in dataset.items():
    code = json.loads(arch)
    if code not in search_space:
        search_space.append(code)

with open('search_space_pretrain', 'wb') as file:
    pickle.dump(search_space, file)

# j = 0
# k = 0
# for i in search_space:
#     if i not in search_space_1:
#         search_space_1.append(i)
#     k+=1
#     print(k)

# search_space = search_space.append(search_space_1)



# with open('search_space_shuffle', 'wb') as file:
#     pickle.dump(search_space, file)