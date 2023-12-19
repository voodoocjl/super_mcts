import csv
import pickle
import os

name = 'mosi_dataset_tq'
dir_path = os.path.dirname(os.path.realpath(__file__))
files = os.listdir(dir_path)
dataset_file = os.path.join(dir_path, name)

if os.path.isfile(dataset_file) == True:
    with open(dataset_file, 'rb') as file:
        dataset = pickle.load(file)
        print("size:", len(dataset))
else:
    dataset = {}

for file in files:
    extension = os.path.splitext(file)[1]
    if extension == '.csv':
        csv_reader = csv.reader(open(os.path.join(dir_path, file)))
        arch_code, energy = [], []
        for row in csv_reader:
            arch_code.append(row[2])
            energy.append(row[4])
        try:
            assert arch_code[0] == 'arch_code' and energy[0] == 'test_mae'
        except AssertionError:
            print(file, 'is a wrong csv files')
            continue
        arch_code.pop(0)
        energy.pop(0)

        for i in range(len(arch_code)):
            if arch_code[i] not in dataset:
                dataset[arch_code[i]] = eval(energy[i])

with open(dataset_file, 'wb') as file:
    pickle.dump(dataset, file)

print("size:", len(dataset))

# with open('dataset.csv', 'a+', newline='') as res:
#     writer = csv.writer(res)

#     for keys in dataset:
#         writer.writerow([1, keys, 0, dataset[keys]])