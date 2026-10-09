datasets = []
with open('data.csv', 'r') as f:
    f.readline()
    for line in f:
        datasets.append(line.strip().split(","))
for i in range(len(datasets)):
    for j in range(5):
        datasets[i][j] = float(datasets[i][j])

print(datasets)

with open('results.txt', 'w') as f:
    f.write("hi\n")
    f.write("hi")