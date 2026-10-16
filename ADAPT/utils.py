import json, pickle, time

def load_text(path:str):
    with open(path) as f:
        return f.read()
def save_text(string, path:str):
    string = str(string)
    with open(path, 'w') as f:
        f.write(string)
def load_json(path:str):
    with open(path) as f:
        return json.load(f)
def save_json(obj, path:str, indent=None):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=indent)
def load_pickle(path:str):
    with open(path, 'rb') as f:
        return pickle.load(f)
def save_pickle(obj, path:str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def show_time(string='', *args, end='\n'):
    local_time = time.localtime()
    formatted_time = time.strftime("[%Y-%m-%d %H:%M:%S]", local_time)
    print(formatted_time + ' '+ string, *args, end=end)
    return formatted_time