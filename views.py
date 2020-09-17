from django.shortcuts import render
from newsletter.models import News,New_News
from django.contrib import auth
import pandas as pd
import numpy as np
import re
import json

from tensorflow.keras.models import load_model
from django_pandas.io import read_frame
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Flatten, Concatenate, Input, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
# Create your views here.
def cleansing(times_data) : 

        # times_data =pd.read_csv(csv_data_path)
    times_data['title'] = times_data['title'].str.replace("[^\w]", " ")

    # 기사 내용 전처리, 괄호 단어 뽑기, 괄호 제거 후 띄어쓰기
    p = re.compile(r'<.+?>') #html 구조 제거
    p2 = re.compile(r'\(([^)]+)') # 괄호 뽑기
    p3 = re.compile( r'\([^)]*\)') # 괄호 제거

    times_data['regex_content'] = ''
    times_data['regex_blank'] = ''

    for n in range(len(times_data['content'])):
        sub_content= re.sub(p,'',times_data['content'][n]) #html 구조 제거한 기사 문장
        times_data['regex_blank'][n]= p2.findall(times_data['content'][n]) #괄호 단어 뽑은 리스트
        sub_content = re.sub(p3,' ',sub_content) #괄호 제거한 기사 문장
        sub_content = sub_content.replace("[^\w]", " ")
        times_data['regex_content'][n] = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ·∙!』\\’‘|\(\)\[\]\<\>`\'…》]', ' ', sub_content)


    # 태그 내용 전처리
    times_data['tag_1'] = ''
    for i in range(len(times_data['tag'])) :
        if times_data['tag'].isnull()[i]: times_data['tag_1'][i] = ""
        else : times_data['tag_1'][i]= times_data['tag'][i].replace(',',' ')

        
    times_data['headline']=times_data['title'] +' ' + times_data['tag_1'] + ' ' + times_data['regex_content']



    # 불용어 제거, 토큰화 진행 --> 띄어쓰기 기준으로 문장 잘라 list에 담기

    stopwords = ['','함께','하지만','뿐','한','또','수','결국','를','을','등','으로','것','약','가','이','즉','은','될','큰','는','로','및','에','그','곧','기자','chosunbiz','며','우리','com','위해','아니라','고','바','와','과','있다','통해','뒤','해','밖에','대한','보다','하는','위한','등을']

    X_token = []
    for stc in times_data['headline']:
        token = []
        words = stc.split()
        for word in words:
            if word not in stopwords:
                token.append(word)
        X_token.append(token)

    return X_token


def news_open(request):
    news_data = News.objects.values()
    new_news_data = New_News.objects.values()
    
    # news_id = News.objects.values_list('id','tag','')
    df0 = read_frame(news_data)
    df1 = read_frame(new_news_data)

# id, tag, content
    news_data=df0.loc[
        :,
        [
            'link',
            'title',
            'date',
            'content',
            'tag',
            'big_category',
            'small_category',
        ]
    ]

    new_news_data=df1.loc[
        :,
        [
            'tag',
            'content',
            'title',
        ]
    ]
# times_data = new_news_data
    
    X_token = cleansing(times_data)
    tokenizer = Tokenizer(150000) 
    category_list = ['기술', '공통', '문화/예술', '경제', '사회', '건강']

    with open(r'C:\Users\leeso\Downloads\wordIndex.json') as json_file:
        word_index = json.load(json_file)
        tokenizer.word_index = word_index

    model = load_model(r'C:\Users\leeso\Downloads\news_classification.h5')
    empty = []
    for token_stc in X_token : 
        encode_stc = tokenizer.texts_to_sequences([token_stc])
        pad_stc = pad_sequences(encode_stc, maxlen=500)
        score = model.predict(pad_stc)
        result_category = category_list[score.argmax()]
        result_prob= score[0, score.argmax()] # 확률값 반환
        empty.append([result_category, result_prob])


    # times_data=news_data
    result=pd.DataFrame(empty, columns=['category','prob'])
    resultttt=pd.concat([times_data,result],axis=1)
    resultttt[['title','category','prob']]

    # context = {'list' : news_data}
 



    return render(request,'newss.html',context)



# def newsletter_modeling(X_token) : 
    

    # context = {
    #     "":,
    #     "":,
    #     "":,
    # }

# #################################################################################################################################

#     resultttt[resultttt.category == '문화/예술'][['tag','headline','category','prob']].sort_values(['prob'],ascending=False) #출력부분
#     return render(request,'newss.html',X_token)