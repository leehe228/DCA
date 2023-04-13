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


@csrf_exempt
def search(request):
    keyword = request.GET['keyword'].strip()

    print("KEYWORD:", parse.unquote(keyword))

    # "dokdo is korean territory"
    if "dokdo" in keyword:
        flag = 0
        proposition = "dokdo is korean territory"
        negation = "takeshima is japanese territory"
    
    # "drinking alcohol is bad for memory"
    elif "memory" in keyword:
        flag = 1
        proposition = "drinking alcohol is bad for memory"
        negation = "drinking alcohol is good for memory"

    # "air quality in korea is bad"
    elif "air" in keyword:
        flag = 2
        proposition = "air quality in korea is bad"
        negation = "air quality in korea is good"
    
    else:
        flag = -1
    
    datas = list()
    con_datas = list()
    data_warning = ""
    con_data_warning = ""

    if flag >= 0:

        print("negation is :", negation)

        # proposition = keyword

        page_url_list = search_google(proposition, negation, flag)

        article_list = sumy(page_url_list)

        summary_list = summarize(article_list)

        similarity_list = evaluate_similarity(proposition, negation, page_url_list, summary_list)
        similarity_list = categorize_by_threshold(similarity_list, .6)

        pros_list, cons_list = split_pros_cons(similarity_list)

        pros_list.sort(key=lambda x : -x['similarity'])
        cons_list.sort(key=lambda x : x['similarity'])

        for pros_dict in pros_list:

            s = Searching(pros_dict['sentence'], pros_dict['page_url'], int(pros_dict['similarity'] * 100))
            datas.append(s)
        
        for cons_dict in cons_list:
            s2 = Searching(cons_dict['sentence'], cons_dict['page_url'], -int(cons_dict['similarity'] * 100))
            con_datas.append(s2)
        
        print("num of pros:", len(datas), " / num of cons:", len(con_datas))

        if len(datas) == 0:
            data_warning = "검색 결과가 없습니다."
        if len(con_datas) == 0:
            con_data_warning = "검색 결과가 없습니다."

    return render(request, 'api/search.html', {'datas':datas, 'data_warning':data_warning, 'con_datas':con_datas, 'con_data_warning':con_data_warning, 'keyword':keyword})

@csrf_exempt
def home(request):
    return render(request, 'api/main.html', {})


def search_google(proposition, negation, flag) -> list:
    search_prop_url = "https://www.google.com/search?q=" + parse.quote(proposition)
    search_negn_url = "https://www.google.com/search?q=" + parse.quote(negation)
    print("search proposition url :", search_prop_url)
    print("search negation url :", search_negn_url)
    print("Searching on Google News...")

    sleep(3)

    if flag == 0:

        prop_page_url_set = {
            "https://en.yna.co.kr/view/AEN20230222006700325",
            "https://koreajoongangdaily.joins.com/2022/12/18/national/diplomacy/Korea-Japan-national-security-strategy/20221218173522005.html",
            "https://asianews.network/dokdo-still-unresolved-as-urgency-mounts/",
            "https://www.asahi.com/ajw/articles/14762186",
            "https://www.korea.net/Government/Current-Affairs/National-Affairs?affairId=83",
            "https://www.lowyinstitute.org/the-interpreter/islands-ire-south-korea-japan-dispute",
            "https://www.nationalgeographic.com/travel/article/history-dispute-photos-dodko-rocks-islands",
            "https://japan-forward.com/editorial-japanese-government-must-show-enough-resolve-to-recover-takeshima/",
            "https://koreajoongangdaily.joins.com/2022/10/13/opinion/columns/light-aircraft-carrier-Korea-Japan/20221013194956483.html",
            "https://www.telesurenglish.net/news/South-Korea-Exercises-Military-Drills-Near-the-Dokdo-Islands-20220729-0019.html"
        }

        negn_page_url_set = {
            "https://japan-forward.com/u-s-air-force-charts-confirm-takeshima-is-japanese-territory/",
            "https://www.japantimes.co.jp/news/2023/02/19/national/politics-diplomacy/japanese-interested-takeshima-islets/",
            "https://www.asahi.com/ajw/articles/14762186",
            "https://japannews.yomiuri.co.jp/editorial/yomiuri-editorial/20230222-92787/",
            "https://thediplomat.com/2021/06/south-korea-erupts-in-outrage-over-tokyo-olympics-map/",
            "https://www.lowyinstitute.org/the-interpreter/islands-ire-south-korea-japan-dispute",
            "https://www.nationalgeographic.com/travel/article/history-dispute-photos-dodko-rocks-islands",
            "https://www.dw.com/en/156-year-old-map-may-reignite-japan-south-korea-island-dispute/a-39966162",
            "https://mainichi.jp/english/articles/20230222/p2g/00m/0na/051000c",
            "https://www.scmp.com/week-asia/politics/article/3179989/seoul-mission-how-marine-survey-disputed-waters-affecting-south"
        }

    elif flag == 1:
        prop_page_url_set = {
            "https://www.newsweek.com/tried-tested-quit-alcohol-dry-january-drinking-memory-1778085",
            "https://www.psypost.org/2022/09/a-moderate-dose-of-alcohol-impairs-the-ability-to-imagine-a-possible-future-situation-63917",
            "https://www.discovermagazine.com/health/this-is-your-brain-off-alcohol",
            "https://www.theatlantic.com/health/archive/2017/12/even-small-amounts-of-alcohol-impair-memory/548474/",
            "https://edition.cnn.com/2021/05/19/health/alcohol-brain-health-intl-scli-wellness/index.html",
            "https://www.health.harvard.edu/blog/this-is-your-brain-on-alcohol-2017071412000",
            "https://www.newsweek.com/what-binge-drinking-does-brain-1769310",
            "https://www.theguardian.com/society/2021/may/18/any-amount-of-alcohol-consumption-harmful-to-the-brain-finds-study",
            "https://www.ajc.com/news/world/want-improve-your-memory-have-drink-after-studying/QyOirxDIehvlWrdtOvB19I/",
            "https://www.usatoday.com/story/news/health/2022/03/09/beer-glass-wine-daily-brain-shrink/9425508002/"
        }

        negn_page_url_set = {
            "https://health.clevelandclinic.org/brownout-vs-blackout/",
            "https://www.eatingwell.com/article/8035262/how-does-alcohol-affect-your-brain-health/",
            "https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1004039",
            "https://www.discovermagazine.com/health/this-is-how-alcohol-affects-the-brain",
            "https://www.washingtonpost.com/wellness/2022/12/29/alcohol-tipsy-brain-social-impacts/",
            "https://scitechdaily.com/even-moderate-drinking-found-to-be-linked-to-brain-changes-and-cognitive-decline/",
            "https://www.medicalnewstoday.com/articles/drinking-just-3-cans-of-beer-a-week-may-be-linked-to-cognitive-decline",
            "https://www.theguardian.com/science/2022/aug/17/stop-drinking-keep-reading-look-after-your-hearing-a-neurologists-tips-for-fighting-memory-loss-and-alzheimers",
            "https://www.yahoo.com/lifestyle/drinking-habits-age-brain-faster-233751840.html",
            "https://www.irishexaminer.com/lifestyle/healthandwellbeing/arid-41046992.html"
        }

    elif flag == 2:

        prop_page_url_set = {
            "https://en.yna.co.kr/view/AEN20230323002800315",
            "https://koreajoongangdaily.joins.com/2023/03/23/national/socialAffairs/korea-dust-fine-dust/20230323183056909.html",
            "https://www.koreatimes.co.kr/www/nation/2023/01/371_343155.html",
            "https://www.scientificamerican.com/article/what-air-pollution-in-south-korea-can-teach-the-world-about-misinformation/",
            "https://koreajoongangdaily.joins.com/2023/01/08/national/socialAffairs/korea-dust-fine-dust/20230108181438507.html",
            "https://www.bbc.com/news/world-asia-48346344",
            "https://www.koreatimes.co.kr/www/nation/2023/01/113_343267.html",
            "https://www.stripes.com/theaters/asia_pacific/army-allows-soldiers-to-wear-masks-while-in-uniform-when-air-quality-is-poor-in-s-korea-1.575182",
            "https://www.voanews.com/a/south-korea-air-pollution/4764898.html",
            "https://en.yna.co.kr/view/AEN20190306009751325"
        }

        negn_page_url_set = {
            "https://en.yna.co.kr/view/AEN20230323002800315",
            "https://koreajoongangdaily.joins.com/2023/03/23/national/socialAffairs/korea-dust-fine-dust/20230323183056909.html",
            "https://www.koreatimes.co.kr/www/nation/2023/01/371_343155.html",
            "https://www.scientificamerican.com/article/what-air-pollution-in-south-korea-can-teach-the-world-about-misinformation/",
            "https://koreajoongangdaily.joins.com/2023/01/08/national/socialAffairs/korea-dust-fine-dust/20230108181438507.html",
            "https://www.straitstimes.com/asia/east-asia/sandstorms-dangerous-pollution-return-to-beijing",
            "https://phys.org/news/2023-02-china-pollution-policies-air-quality.html",
            "https://en.yna.co.kr/view/AEN20221201009800325",
            "https://www.koreatimes.co.kr/www/nation/2023/01/113_343267.html",
            "https://www.bbc.com/news/world-asia-48346344"
        }

    page_url_list = list(prop_page_url_set | negn_page_url_set)

    for i, url in enumerate(page_url_list):
        print(f"url #{i + 1}\n{url}")
        sleep(1)

    return page_url_list


from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

def sumy(page_url_list):
    LANGUAGE = "english"
    SENTENCES_COUNT = 10
    article_list = []
    
    for url in page_url_list:
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
            
        article_list.append(article)
        
    print(len(article_list), "articles collected from Google News.")

    return article_list

from transformers import pipeline
from transformers import AutoTokenizer


def count_token(article):
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
    tokenized_input = tokenizer(article, truncation=True, padding=True)
    
    return len(tokenized_input['input_ids'])

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize(article_list):
    summary_list = []

    for i, article in enumerate(article_list):
        print(f"summary #{i + 1}")
        num_token = count_token(article)
        summ = summarizer(article, max_length=num_token // 2, min_length=num_token // 5, do_sample=False)
        summary_list.append(summ[0]['summary_text'])
        print(summ[0]['summary_text'])
        print()
                          
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