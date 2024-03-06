import csv

#read csv file
with open("static.csv") as f:
    reader = csv.reader(f)
    data = [row for row in reader]

#modify data
for i in range(len(data)):
    data[i] = data[i] + [0] * 42

#write to new file
with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(data)