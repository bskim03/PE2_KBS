# 2022047810 김범서

import pandas as pd

# Set column names
cols = ['col1', 'col2', 'col3']

# Create 2D list
list2 = [[1, 2, 3], [11, 12, 13]]

# Create DataFrame
df_list2 = pd.DataFrame(list2, columns=cols)
print(df_list2)
