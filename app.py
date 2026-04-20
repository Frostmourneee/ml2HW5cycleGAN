import os
import requests
import zipfile

dataset_folder = "./apple2orange_data"
os.makedirs(dataset_folder, exist_ok=True)

url = "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/apple2orange.zip"
zip_path = os.path.join(dataset_folder, "apple2orange.zip")

print("Downloading...")
response = requests.get(url)
with open(zip_path, "wb") as f:
    f.write(response.content)

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_folder)

os.remove(zip_path)

print(f"Done! Files in: {dataset_folder}/apple2orange/")
print("Folders: trainA, trainB, testA, testB")