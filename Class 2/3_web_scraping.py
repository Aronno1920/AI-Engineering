import requests
from bs4 import BeautifulSoup
import json


# Step 1: Fetch the page
url = "http://quotes.toscrape.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")


# Step 2: Extract quotes and authors
quotes_data = []
quotes = soup.find_all("div", class_="quote")
for quote in quotes:
    text = quote.find("span", class_="text").text.strip()
    author = quote.find("small", class_="author").text.strip()
    quotes_data.append({"quote": text, "author": author})


# Step 3: Save to JSON
with open("quotes.json", "w", encoding="utf-8") as json_file:
    json.dump(quotes_data, json_file, indent=4, ensure_ascii=False)
    print(" Saved to quotes.json")
