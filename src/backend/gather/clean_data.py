import os
import csv
from time import sleep

# this script serves to prune data that does not fit the required parameters
# e.g.
# static data should be 64 columns long
# one hand dynamic data should be 1891 columns long 
# two hand dynamic data should be 3781 columns long

cur_dir = os.curdir

# dataset number: (file path, length of expected data)
datasets = {'0': ([cur_dir + '/datasets/static.csv', 64]),
            '1': ([cur_dir + '/datasets/dynamic.csv', 1891]),
            '2': ([cur_dir + '/datasets/dynamic_two_1.csv', cur_dir + '/datasets/dynamic_two_2.csv'], 1891),
            # '9': (cur_dir + '/datasets/dynamic_2 copy.csv', 3781) used for testing
            }

sleep(0.1)

t = input("Enter dataset to clean: static (0), one hand dynamic (1), two hand dynamic (2), or all datasets (3)\n")

if t == '3':
    #TODO 
    pass
else:
    for s in datasets[t][0]:
        with open(s, 'r', encoding="UTF8", newline='') as f:
            reader = csv.reader(f)
            filtered_rows = []

            for row in reader:
                row_str = ','.join(row).rstrip(', \n\r')
                clean_row = row_str.split(',')
                if len(clean_row) == datasets[t][1]:
                    filtered_rows.append(clean_row)
        
        with open(s, 'w', encoding="UTF8", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(filtered_rows)