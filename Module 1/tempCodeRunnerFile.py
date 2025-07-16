import csv

### how to read csv
with open('products.csv','r',newline='') as r_csv:
    reader = csv.reader(r_csv)
    for row in reader:
        print(row)