Explore data  
[1] 查看urls 在不同情感的的数据中的分布情况
在数据的'selected_text' 字段中查找包含url的数据，并根据情感标签分类计数。  
在 中立情感中的selected_text中url出现的概率要远大其他两类情感。  
![image1](https://github.com/Hanshawn11/Kaggle/blob/master/TweetSentiment/EDA/images/urls%20(1).png)  


[2] 查看'selected_text'字段中单词数量分布情况。从左到右依次为negative - positive - neutral。  selected_text文本的最大单词数量是35， 大多数都集中在5个单词以下，设置encode的长度时可以设定为35个单词， 如果是按字符编码可以设定为120左右。  
    
  ![word_counts](https://raw.githubusercontent.com/Hanshawn11/Kaggle/master/TweetSentiment/EDA/images/word_counts.bmp)
  
[3] 统计在各类情感中出现次数最多的单词。  在各类情感中的词语分布情况基本符合自然的猜想。 
  ![word_clouds](https://raw.githubusercontent.com/Hanshawn11/Kaggle/master/TweetSentiment/EDA/images/word_cloud.bmp)
  
 
[4] 查看各类情感在训练数据和测试数据中的分布情况， 分布情况几乎均衡， 做交叉验证时应该保证分布情况和训练数据的分布情况一致。  
![distribute](https://github.com/Hanshawn11/Kaggle/blob/master/TweetSentiment/EDA/images/senti%20distribute.png)
