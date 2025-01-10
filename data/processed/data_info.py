import pandas as pd  
import json  
from pathlib import Path  
import matplotlib.pyplot as plt  
import seaborn as sns  

# 设置显示选项  
#pd.set_option('display.max_columns', None)  # 显示所有列  
#pd.set_option('display.width', None)       # 加宽显示宽度  
#pd.set_option('display.max_rows', 10)      # 显示的最大行数  
#pd.set_option('display.max_colwidth', None)  # 显示完整的列内容

# 读取JSON文件  
def load_json_file(file_path):  
    data = []  
    with open(file_path, 'r', encoding='utf-8') as f:  
        for line in f:  
            data.append(json.loads(line))  
    return pd.DataFrame(data)  

# 读取数据  
data_dir = Path('data/processed')  
item_df = load_json_file('item.json')  
review_df = load_json_file('review.json')  
user_df = load_json_file('user.json')  

# 基本信息概览  
print("=== Dataset Overview ===")  
print(f"Number of Items: {len(item_df)}")  
print(f"Number of Reviews: {len(review_df)}")  
print(f"Number of Users: {len(user_df)}")  

# 显示每个DataFrame的基本信息  
print("\n=== Items Data Overview ===")  
print(item_df.info())  
print("\nSample data:")  
print(item_df.head(2))  

print("\n=== Reviews Data Overview ===")  
print(review_df.info())  
print("\nSample data:")  
print(review_df.head(2))  

print("\n=== Users Data Overview ===")  
print(user_df.info())  
print("\nSample data:")  
print(user_df.head(2))  

# 基本统计  
print("\n=== Basic Statistics ===")  
print("\nItems rating distribution:")  
print(item_df['stars'].describe())  