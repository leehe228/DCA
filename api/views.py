from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import auth
from django.views.decorators.csrf import csrf_exempt

from urllib import parse
from time import sleep

# Create your views here.

class Searching(object):
    def __init__(self, sentence, page_url, similarity):
        self.page_url = page_url
        self.similarity = similarity
        self.sentence = sentence

        color = "#ffffff"
        if similarity >= 70:
            color = "#09831d"
        elif similarity >= 65:
            color = "#e97f28"
        else:
            color = "#db2020"

        self.bar_color = color

    def __str__(self):
        return {'sentence':self.sentence, 'similarity':self.similarity, 'page_url':self.page_url, 'bar_color':self.color}


import googletrans

translator = googletrans.Translator()

def trans(sentence, dest):
    return translator.translate(sentence, dest=dest).text


from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

def convert2negation(sentence):
    tagged = pos_tag(word_tokenize(sentence)) 
    
    tokenized = word_tokenize(sentence)
    
    for i, (word, tag) in enumerate(tagged[::-1]):
        if "VB" in tag:
            tokenized[len(tagged) - i - 1] = f"not {tokenized[len(tagged) - i - 1]}"
            break

    negation = " ".join(tokenized)
    
    return negation


from langdetect import detect

@csrf_exempt
def search(request):
    keyword = parse.unquote(request.GET['keyword'].strip())

    print("KEYWORD:", keyword, detect(keyword))

    try:
        if detect(keyword) == 'ko':
            proposition = trans(keyword, 'en')
            neg = convert2negation(proposition)
            negation = trans(trans(neg, 'ko'), 'en')

            print("proposition :", proposition)
            print("negation :", negation)

            flag = 0

        else:
            proposition = keyword
            neg = convert2negation(proposition)
            negation = trans(trans(neg, 'ko'), 'en')

            print("proposition :", proposition)
            print("negation :", negation)

            flag = 0

        
    except Exception as e:
        print(e)
        flag = -1
    
    datas = list()
    con_datas = list()
    data_warning = "검색 결과가 없습니다."
    con_data_warning = "검색 결과가 없습니다."

    if flag > -1:

        page_url_list = search_google(proposition, negation)

        article_list = sumy(page_url_list, count=10)

        summary_list = summarize(article_list, count=10)

        similarity_list = evaluate_similarity(proposition, negation, page_url_list, summary_list)
        similarity_list = categorize_by_threshold(similarity_list, .6)

        pros_list, cons_list = split_pros_cons(similarity_list)

        pros_list.sort(key=lambda x : -x['similarity'])
        cons_list.sort(key=lambda x : x['similarity'])

        for pros_dict in pros_list:

            s = Searching(trans(pros_dict['sentence'], dest='ko'), pros_dict['page_url'], int(pros_dict['similarity'] * 100))
            datas.append(s)
        
        for cons_dict in cons_list:
            s2 = Searching(trans(cons_dict['sentence'], dest='ko'), cons_dict['page_url'], -int(cons_dict['similarity'] * 100))
            con_datas.append(s2)
    
    print("num of pros:", len(datas), " / num of cons:", len(con_datas))

    if len(datas) > 0:
        data_warning = ""
    if len(con_datas) > 0:
        con_data_warning = ""

    return render(request, 'api/search.html', {'datas':datas, 'data_warning':data_warning, 'con_datas':con_datas, 'con_data_warning':con_data_warning, 'keyword':keyword})

@csrf_exempt
def home(request):
    return render(request, 'api/main.html', {})

import requests 
import lxml
from bs4 import BeautifulSoup as bs


def search_google(proposition, negation) -> list:
    #pros_params1 = {'q' : proposition , 'hl' : 'ko', 'tbm' : 'nws', 'start' : '0'}
    #pros_params2 = {'q' : proposition , 'hl' : 'ko', 'tbm' : 'nws', 'start' : '10'}
    #cons_params1 = {'q' : negation , 'hl' : 'ko', 'tbm' : 'nws', 'start' : '0'}
    #cons_params2 = {'q' : negation , 'hl' : 'ko', 'tbm' : 'nws', 'start' : '10'}

    pros_params1 = {'q' : proposition , 'hl' : 'ko', 'start' : '0'}
    pros_params2 = {'q' : proposition , 'hl' : 'ko', 'start' : '10'}
    cons_params1 = {'q' : negation , 'hl' : 'ko', 'start' : '0'}
    cons_params2 = {'q' : negation , 'hl' : 'ko', 'start' : '10'}
    
    pros_params_list = [pros_params1, pros_params2]
    cons_params_list = [cons_params1, cons_params2]
    
    pros_page_url_list = []
    cons_page_url_list = []

    header = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36'} 
    cookie = {'CONSENT' : 'YES'}
    url = 'https://www.google.com/search?'
    
    n = 0
    
    for params in pros_params_list:
        res = requests.get(url, params = params, headers = header, cookies = cookie)
        soup = bs(res.text, 'lxml')

        # l1 = soup.find_all('div', 'mCBkyc ynAwRc MBeuO nDgy9d')
        # l2 = soup.find_all('a', 'WlydOe')
        l1 = soup.select("div.yuRUbf > a")
        
        """for i, j in zip(l1, l2):
            # print(i.get_text())
            print("URL ", n, "\n", j.get('href'))
            pros_page_url_list.append((n, j.get('href')))
            n += 1
        """
        for i in l1:
            print("URL", n + 1, "\n", i.get("href"))
            pros_page_url_list.append((n, i.get('href')))
            n += 1
    
    n = 0
    
    for params in cons_params_list:
        res = requests.get(url, params = params, headers = header, cookies = cookie)
        soup = bs(res.text, 'lxml')

        # l1 = soup.find_all('div', 'mCBkyc ynAwRc MBeuO nDgy9d')
        # l2 = soup.find_all('a', 'WlydOe')
        l1 = soup.select("div.yuRUbf > a")

        """for i, j in zip(l1, l2):
            # print(i.get_text())
            print("URL ", n, "\n", j.get('href'))
            
            if j.get('href') in pros_page_url_list:
                pass
            else:
                cons_page_url_list.append((n, j.get('href')))
                n += 1
        """
        for i in l1:
            if i.get('href') in pros_page_url_list:
                pass
            else:
                print("URL", n + 1, "\n", i.get("href"))
                cons_page_url_list.append((n, i.get('href')))
            
            n += 1
                
    print("pros :", len(pros_page_url_list), ", cons :", len(cons_page_url_list))
    
    url_list = pros_page_url_list + cons_page_url_list
    url_list.sort(key=lambda x : x[0])
    url_list = list(map(lambda x : x[1], url_list))
    
    return url_list


from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def sumy(page_url_list,  count):
    LANGUAGE = "english"
    SENTENCES_COUNT = 5
    article_list = []

    print("collecting articles from server...")

    for n, url in enumerate(page_url_list):
        print("collect", n + 1, "th article")
        try:
            parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
            # or for plain text files
            # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
            # parser = PlaintextParser.from_string("Check this out.", Tokenizer(LANGUAGE))
            stemmer = Stemmer(LANGUAGE)
            
            summarizer = Summarizer(stemmer)
            summarizer.stop_words = get_stop_words(LANGUAGE)
            
            article = ""
            for i, sentence in enumerate(summarizer(parser.document, SENTENCES_COUNT)):
                article += str(sentence)
            
            if len(article.strip()) > 0:
                article_list.append(article)

        except Exception as e:
            print(e)

        if len(article_list) >= count:
            break
        
    print(len(article_list), "articles collected from Google News.")

    return article_list

from transformers import pipeline
from transformers import AutoTokenizer


def count_token(article):
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
    tokenized_input = tokenizer(article, truncation=True, padding=True)
    
    return len(tokenized_input['input_ids'])

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(article_list, count):
    summary_list = []

    for i, article in enumerate(article_list):
        print(f"summary #{i + 1}")
        num_token = count_token(article)
        try:
            summ = summarizer(article, max_length=num_token // 2, min_length=num_token // 5, do_sample=False)
            summary_list.append(summ[0]['summary_text'])
            print(summ[0]['summary_text'])
            print()
        except Exception as e:
            print(e)
            print(f"summary #{i + 1} passed because of an error.")
                          
    print(len(summary_list), "article summarized.")

    return summary_list


def split_to_sentences(summary_list):
    for i, summary in enumerate(summary_list):
        print(i)
        for sentence in summary.split(". "):
            print("- " + sentence)
        print("-"*80, "\n\n")


from sentence_transformers import SentenceTransformer, util
import scipy


summarizer_dict = {}
summarizer_dict['model'] = SentenceTransformer('roberta-large-nli-mean-tokens')

def evaluate_similarity(proposition, negation, page_url_list, summary_list):
    similarity_list = []
    
    v_proposition = summarizer_dict['model'].encode(proposition)
    v_negation = summarizer_dict['model'].encode(negation)
    
    for i, summary in enumerate(summary_list):
        summary_dict = {'index':i, 'sentences':[], 'page_url':page_url_list[i]}
        value_list = []
        for sentence in summary.split(". "):
            d = {}
            v_sentence = summarizer_dict['model'].encode(sentence)
            pro_cos_sim = float(util.pytorch_cos_sim(v_proposition, v_sentence))
            con_cos_sim = -float(util.pytorch_cos_sim(v_negation, v_sentence))
            d['sentence'] = sentence
            d['pro_cos_sim'] = pro_cos_sim
            d['con_cos_sim'] = con_cos_sim
            summary_dict['sentences'].append(d)
        if len(value_list) > 0:
            similarity = sum(value_list) / len(value_list)
            summary_dict['similarity'] = similarity
        else:
            summary_dict['similarity'] = 0.
        similarity_list.append(summary_dict)

    return similarity_list


def categorize_by_threshold(similarity_list, threshold=.70):
    for summary_dict in similarity_list:
        for sentence_dict in summary_dict['sentences']:
            pro_cos_sim = sentence_dict['pro_cos_sim']
            con_cos_sim = sentence_dict['con_cos_sim']
            # pros
            if pro_cos_sim > -con_cos_sim and pro_cos_sim >= threshold:
                flag = "pros"
            # cons
            elif pro_cos_sim < -con_cos_sim and -con_cos_sim >= threshold:
                flag = "cons"
            else:
                flag = "neut"
            sentence_dict['flag'] = flag

    return similarity_list


def split_pros_cons(similarity_list):
    pros_list = []
    cons_list = []
    pros_similarity = 0.
    cons_similarity = 0.

    for x in similarity_list:
        for sentence in x['sentences']:
            if sentence['flag'] == "pros":
                pros_list.append({'sentence':sentence['sentence'], 'similarity':sentence['pro_cos_sim'], 'article_index':x['index'], 'page_url':x['page_url'], 'article_similarity':x['similarity']})

            elif sentence['flag'] == "cons":
                cons_list.append({'sentence':sentence['sentence'], 'similarity':sentence['con_cos_sim'], 'article_index':x['index'], 'page_url':x['page_url'], 'article_similarity':x['similarity']})
                
    return pros_list, cons_list


def print_pros(proposition, pros_list):
    print('-' * 80)
    print(f"[PROPOSITION] {proposition}\n{len(pros_list)} sources found!")
    print('-' * 80)
    for pros_dict in pros_list:
        print(f" - {pros_dict['sentence']}")
        print(f" - similarity : {pros_dict['similarity']}")
        print(f"{pros_dict['page_url']}")
        print()


def print_cons(negation, cons_list):
    print('-' * 80)
    print(f"[NEGATION] {negation}\n{len(cons_list)} sources found!")
    print('-' * 80)
    for cons_dict in cons_list:
        print(f" - {cons_dict['sentence']}")
        print(f" - similarity : {cons_dict['similarity']}")
        print(f"{cons_dict['page_url']}")
        print()