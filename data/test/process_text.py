import csv
import re
import sys
import random
csv.field_size_limit(sys.maxsize)

with open('train.txt', 'w') as f:
    with open('zhihu_answer_20181107.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            # tmp = random.randint(1, 100)
            # if(tmp != 1):
            #     continue
            if(i % 500 !=0):
                continue
            content = row['content']
            content = re.sub(r'<[^>]*>', '', content)
            if (len(content) < 300):
                continue
            f.write(content)



