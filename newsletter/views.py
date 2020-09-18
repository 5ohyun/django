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

    stopwords = ['','밝혔다','있는','의','관련','예정이다','주로','이를','보면','두고','줄','게','역시', '각','볼','다', '등이다','수도','매우','중요한','보였다','혹은','등과','이라는', '관한','itchosun','한다고','이어','후','매일', '여부를','등은','이들은','그동안','했다','할', '◇','시','모든','현재', '1','주요', '이후','설명했다','전','경우','내','하지만','그는', '같은', '총','따라', '가장','것이','대비','관계자는','있도록','최근','기존','한다','것이다','라고','더','중','따르면', '다양한','말했다','이번','것으로','한','이에','다만','하고','또','함께','수','를','을','등','으로','것','약','가','이','즉','은','될','큰','는','로','및','에','그','곧','기자','chosunbiz','며','우리','com','위해','아니라','고','바','와','과','있다','통해','뒤','해','밖에','대한','보다','하는','위한','등을']

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
    news_data=df0.loc[:,
        [
            'link',
            'title',
            'date',
            'content',
            'tag',
            'team_category',
        ]]

    new_news_data=df1.loc[:,
        [
            'link',
            'tag',
            'title',
            'content',
            'date',
            'big_category'
        ]]
# times_data = new_news_data
    
    X_token = cleansing(new_news_data)
    tokenizer = Tokenizer(150000) 
    category_list = ['기술', '공통', '문화/예술', '경제', '사회', '건강']

    with open(r'C:\Users\leeso\Downloads\wordIndex_0918.json') as json_file:
        word_index = json.load(json_file)
        tokenizer.word_index = word_index

    model = load_model(r'C:\Users\leeso\Downloads\news_classification_128_500_10.h5')
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
    result_append=pd.concat([new_news_data,result],axis=1)

    aitimes = result_append.loc[:,['link','tag','title','content','date','category']]

    news_data = news_data.rename(columns= {'team_category' : 'category'})

    resultttt=pd.concat([aitimes, news_data],axis=0)
    resultttt.reset_index(drop=True, inplace=True)

#     return resultttt

# # ###########################################################################################################################
# # ####################                                     문화/예술                             ############################
# # ###########################################################################################################################
# def news_art_list(request) : 
    # resultttt=news_open()
    resultttt = pd.DataFrame(resultttt)
    art=resultttt[resultttt.category == '문화/예술'][['link','title','tag','category']]
    art.reset_index(drop=True,inplace=True)

    art_link = []
    art_headline = []
    art_tag = []
    art_category = []
    

    for i in range(10) :
        art_link.append(art['link'][i])
        art_tag.append(art['tag'][i])
        art_headline.append(art['title'][i])
        art_category.append(art['category'][i])


    context = {
        'art_link':art_link,
        'art_tag' : art_tag,
        'art_headline' : art_headline,
        'art_category': art_category,

    }
    
    return render(request, "newss.html", context)
        

    
###########################################################################################################################
####################                                         경제                               ############################
###########################################################################################################################

    # economy=resultttt[resultttt.category == '경제'][['link','title','tag','category']]
    # economy.reset_index(drop=True,inplace=True)

    # economy_link = []
    # economy_headline = []
    # economy_tag = []
    # economy_category = []
    

    # for i in range(10) :
    #     economy_link.append(economy['link'][i])
    #     economy_tag.append(economy['tag'][i])
    #     economy_headline.append(economy['title'][i])
    #     economy_category.append(economy['category'][i])


    #     context = {
    #         'economy_link':economy_link,
    #         'economy_tag' : economy_tag,
    #         'economy_headline' : economy_headline,
    #         'economy_category': economy_category,
    #     }
    
#     return render(request, "study_rec/study_rec_list.html", context)

    
# ###########################################################################################################################
# ####################                                        사회                               ############################
# ###########################################################################################################################

#     social=resultttt[resultttt.category == '사회'][['link','title','tag','category']]
#     social.reset_index(drop=True,inplace=True)

#     social_link = []
#     social_headline = []
#     social_tag = []
#     social_category = []
    

#     for i in range(10) :
#         social_link.append(social['link'][i])
#         social_tag.append(social['tag'][i])
#         social_headline.append(social['title'][i])
#         social_category.append(social['category'][i])


#         context = {
#             'social_link':social_link,
#             'social_tag' : social_tag,
#             'social_headline' : social_headline,
#             'social_category': social_category,
#         }

       
# ###########################################################################################################################
# ####################                                         건강                               ############################
# ###########################################################################################################################

#     health=resultttt[resultttt.category == '건강'][['link','title','tag','category']]
#     health.reset_index(drop=True,inplace=True)

#     health_link = []
#     health_headline = []
#     health_tag = []
#     health_category = []
    

#     for i in range(10) :
#         health_link.append(health['link'][i])
#         health_tag.append(health['tag'][i])
#         health_headline.append(health['title'][i])
#         health_category.append(health['category'][i])


#         context = {
#             'health_link':health_link,
#             'health_tag' : health_tag,
#             'health_headline' : health_headline,
#             'health_category': health_category,
#         }


#     return render(request, "study_rec/study_rec_list.html", context)