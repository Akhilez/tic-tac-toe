import json
import os


class DataManager:

    def __init__(self, file_name='data/data.json', max_size=100):
        self.file_name = file_name
        self.max_size = max_size
        self.data = None

    def write(self):
        with open(self.file_name, 'w') as data:
            data.write(json.dumps({'matches': self.data}))

    def enqueue(self, matches):
        if self.data is not None:
            matches.extend(self.data)
        self.data = matches[:self.max_size]

    def get(self):
        if self.data is None:
            if os.path.exists(self.file_name):
                with open(self.file_name, 'r') as data:
                    data_string = data.read()
                    if len(data_string) > 0:
                        self.data = json.loads(data_string)['matches']
            else:
                print(f"Path {self.file_name} does not exist")
        return self.data

    def clear(self):
        self.data = None
        try:
            with open(self.file_name, 'w') as data:
                data.write('')
        except FileNotFoundError:
            return

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        self.write()
