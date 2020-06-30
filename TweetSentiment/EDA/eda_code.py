import matplotlib.pyplot as plt
import pandas as pd
import re

#[1] find urls
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

#[2] words_count
def remove_link(string): 
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'," ",string)
    return " ".join(text.split())

def remove_punct(text):
    line = re.sub(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]+'," ",text)
    return " ".join(line.split())

train['target']=train['selected_text'].apply(lambda x:remove_link(x))
train['target']=train['selected_text'].apply(lambda x:remove_punct(x))
train['target_tweet_length']=train['target'].str.split().map(lambda x: len(x)) # 去除标点符号，urls 统计单词数量

new = train.groupby('sentiment')   # 分组， 分别统计各种情感的单词分布信息
neutral = new.get_group('neutral').reset_index().describe()
positive = new.get_group('positive').reset_index().describe()
negative.get_group('negative').reset_index().describe()
