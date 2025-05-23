import requests

### basic call
response = requests.get('https://jsonplaceholder.typicode.com/todos/1')
print(response.json())


### get api call - with parameters
# url = 'https://jsonplaceholder.typicode.com/posts'
# params = {'id':1}
# response = requests.get(url, params=params)
# print("Status Code:", response.status_code) 
# print("Response JSON:", response.json())


### post api call
# url = "https://jsonplaceholder.typicode.com/posts" 
# data = { "title": "TechAid24", "body": "This is my first POST request!", "userId": 1 } 

# response = requests.post(url, json=data) 
# print("Status Code:", response.status_code) 
# print("Response JSON:", response.json())