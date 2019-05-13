import json
import glob

json_files = glob.glob('poetry_json/*.json')

with open('train.txt', 'w') as w:
    for file in json_files:
        with open(file, 'r') as f:
            str_data = f.read()
            dict_data = json.loads(str_data)
            for i in range(len(dict_data)):
                content = ''
                for j in range(len(dict_data[i]['paragraphs'])):
                    content += dict_data[i]['paragraphs'][j]
                w.write(content+'\n')
