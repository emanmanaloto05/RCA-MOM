# test_langsmith.py

from langsmith import Client

client = Client()

projects = list(client.list_projects())

print("Connected Successfully")
print("Projects Found:", len(projects))