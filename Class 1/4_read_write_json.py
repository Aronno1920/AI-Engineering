import pandas as pd

# Create a sample DataFrame
data = {
    "name": ["Alice", "Bob"],
    "age": [30, 25],
    "city": ["New York", "Los Angeles"]
}
df = pd.DataFrame(data)
df.to_json('data.json', orient='records', indent=4) # Write DataFrame to a JSON file


# Read JSON file into a DataFrame
df = pd.re('data.json')
print(df)  # Display the DataFrame