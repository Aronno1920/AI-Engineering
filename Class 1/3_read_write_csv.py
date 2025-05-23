import csv

### how to read csv
with open('products.csv','r',newline='') as r_csv:
    reader = csv.reader(r_csv)
    for row in reader:
        print(row)

### how to write in csv
data =[
    ['name','brand','price'],
    ['IPhone 16 Pro', 'Apple',210000],
    ['IPhone 15 Pro Max', 'Apple',190000],
    ['Galaxy S25 Ultra', 'Samsung',230000],
    ['Galaxy A74', 'Samsung',74000],
    ['Galaxy A34', 'Samsung',54000]
]

with open('products.csv', 'w', newline='') as w_csv:
    writer = csv.writer(w_csv)
    writer.writerows(data)