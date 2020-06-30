import matplotlib.pyplot as plt
import pandas as pd
import re

train = pd.read_csv('train.csv').fillna('')
patt = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
def find_link(string):
    url = re.findall(patt, string)
    return "".join(url)

train["target"] = train['selected_text'].str.lower()
train['target_url'] = train['target'].apply(lambda x: find_link(x))
df = pd.DataFrame(train.loc[train['target_url'] != '']['sentiment'].value_counts()).reset_index()  #筛选urls字段不为空的数据，并根据sentiment字段分类统计个数
df = df.rename(columns={'index': 'sentiment', 'sentiment':'url_count'})

fig = plt.gcf()
fig.set_size_inches(12, 5.5)   # 设置画布大小
count_list = df['url_count'].values   
count_list = list(arr)     
name_list = list(df['sentiment'].values)
plt.barh(range(3), count_list, color='rgb', tick_label=name_list)
plt.xlabel('counts')
plt.ylabel('sentiment')
plt.show()
plt.savefig('urls.png')
