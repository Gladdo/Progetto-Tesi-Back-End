import requests
import sys

url = 'http://127.0.0.1:8000/diffusers_api/post_lora_model'
data = {'lora_model_name': sys.argv[1], 'username' : sys.argv[2], 'password' : sys.argv[3]}

csrftoken = requests.get(url).cookies['csrftoken']

header = {'X-CSRFToken': csrftoken}
cookies = {'csrftoken': csrftoken}

files = {'lora_model' : open("./training-output/model/" + sys.argv[1] + ".safetensors", 'rb')}

response = requests.post(url, files=files, data=data, headers=header, cookies=cookies)

print(response)

