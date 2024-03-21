# 2022047810 김범서

import pandas as pd

# Set subjects as columns
cols = ['국어', '수학', '영어', '과학', '사회']
# Set students as rows
indexes = ['태현', '준수', '기준']
# Score data
lists = [[83, 68, 92, 55, 85], [40, 95, 64, 87, 77], [65, 87, 58, 92, 72]]

# Set column names using index
dfs = pd.DataFrame(lists, columns=cols, index=indexes)

# Export DataFrame to csv file
# Set encoding for preventing broken letters
dfs.to_csv("./hw10_result.csv", encoding='ANSI')

print(dfs)
