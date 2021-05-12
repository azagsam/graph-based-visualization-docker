import pandas as pd
import os

root = '/home/ales/Documents/RSDO/graph-based-visualization-docker/data/candas_metadata'
for file in os.scandir(root):
    full_path = file.path
    # print(full_path)
    df = pd.read_table(full_path)
    df.rename({'source': 'class'}, inplace=True, axis='columns')
    name = file.name.split('.')[0]
    print(name)
    output = os.path.join(root, name + '.csv')
    df.to_csv(output, index=False)