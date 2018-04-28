import pandas as pd
from collections import defaultdict, Counter

# import input data
topic_file = pd.read_csv("topic.csv")
virality_file = pd.read_csv("virality.csv")

x = Counter(['x','y','x'])
print(x)

