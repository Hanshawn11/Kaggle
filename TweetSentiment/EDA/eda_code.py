import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

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

#[3] word cloud 制作
stop_words = ['on','in', 'at', 'my', 'me', 'you', 'a', 'i', 'the', 'that', 'which', '2', 'm', 's', 'e', 'to', 'it', 'for', 'is', 'and', 'so', 't', 'of']
def create_corpus(data,feature,sentiment):
    #去除stop words 将出现的单词放在一个列表中
    
    corpus=[]
    for x in data[data['sentiment']==sentiment][feature].str.split():
        for i in x:
            if i.lower() not in stop_words:
                corpus.append(i.lower())
    return corpus

pos_words = Counter(create_corpus(train, 'target', 'positive'))   # 利用counter 构建词频字典
pos_words_sorted  = sorted(pos_words.items(), key = lambda x:x[1], reverse=True)[:50]  # 排序并选取出现次数最多top 50 个单词

negat_words = Counter(create_corpus(train, 'target', 'negative'))
negat_words_sorted  = sorted(negat_words.items(), key = lambda x:x[1], reverse=True)[:50]  

neutra_words = Counter(create_corpus(train, 'target', 'neutral'))
neutra_words_sorted  = sorted(neutra_words.items(), key = lambda x:x[1], reverse=True)[:50]  

png = 'twitt.png'  # <- 词云要填充的图案，我这里用推特小鸟
fig = plt.gcf()
fig.set_size_inches(16, 14)

fig1 = plt.subplot(131)  # 1 行 3列 第 1张图
data = {ele[0]:ele[1]  for ele in neutra_words_sorted}  #构建word：count 字典
mask = np.array(Image.open(png))                        # img -> array, 词云的填充对象
wc1 = WordCloud(background_color='white', mask=mask).generate_from_frequencies(data)
plt.imshow(wc1)
plt.title('neutral sentiment')
plt.axis('off')            #关闭边框线

fig2 = plt.subplot(132)
data = {ele[0]:ele[1]  for ele in negat_words_sorted}
mask = np.array(Image.open(png))
wc1 = WordCloud(background_color='white', mask=mask).generate_from_frequencies(data)
plt.imshow(wc1)
plt.title('negative sentiment')
plt.axis('off')

fig3 = plt.subplot(133)
data = {ele[0]:ele[1]  for ele in pos_words_sorted}
mask = np.array(Image.open(png))
wc1 = WordCloud(background_color='white', mask=mask).generate_from_frequencies(data)
plt.imshow(wc1)   # 数字图像显示
plt.title('positive sentiment')
plt.axis('off')
plt.savefig('words_cloud.png')
plt.show()

#饼图
news = train.groupby('sentiment')
pos_data = news.get_group('positive').shape[0]
neg_data =  news.get_group('negative').shape[0]
neutral_data =  news.get_group('neutral').shape[0]

test = pd.read_csv('test.csv')
test_new = test.groupby('sentiment')
pos = test_new.get_group('positive').shape[0]
neg =  test_new.get_group('negative').shape[0]
neutral =  test_new.get_group('neutral').shape[0]

fig = plt.gcf()
fig.set_size_inches(6, 8)
fig1 = plt.subplot(121)

data = [ pos_data, neg_data, neutral_data]     # data 
lables = ['positive', 'negative', 'neutral']  
colors = ['lightgreen', 'grey', 'lightblue']
explode = (0, 0, 0.1)        #设置第三款分割出
plt.pie(data,labels = lables, explode = explode, shadow=True,colors = colors, autopct= '%1.2f%%', pctdistance = 0.5) # pctdistance 距离圆心距离
plt.title('train data')

fig2 = plt.subplot(122)
data = [pos, neg, neutral]
lables = ['positive', 'negative', 'neutral']
colors = ['lightgreen', 'grey', 'lightblue']
explode = (0, 0, 0.1)
plt.pie(data,labels = lables, explode = explode, shadow=True,colors = colors, autopct= '%1.2f%%', pctdistance = 0.5)
plt.title('test data')

plt.savefig('senti distribute.png')
plt.show()
