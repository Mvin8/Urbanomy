import json

def read_json(file_path : str) -> list | dict:
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def write_json(data : dict, file_path : str):
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)