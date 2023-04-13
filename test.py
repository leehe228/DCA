# ==============================
# Convert proposition to negation
# ==============================


from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

def convert2negation(proposition : str) -> str:
    tagged = pos_tag(word_tokenize(proposition)) 
    
    tokenized = word_tokenize(proposition)
    
    for i, (word, tag) in enumerate(tagged[::-1]):
        if "VB" in tag:
            tokenized[len(tagged) - i - 1] = f"not {tokenized[len(tagged) - i - 1]}"
            break

    negation = " ".join(tokenized)
    
    return negation


# negation = convert2negation(proposition)


# ==============================
# Search on Google and collect URLs
# ==============================


from time import sleep

def search(keyword : str) -> list:
    search_url = "https://www.google.com/search?q=" + parse.quote(keyword.strip())
    print("search url :", search_url)
    print("searching...")
    
    sleep(3)
    
    page_url_list = [
        "https://koreajoongangdaily.joins.com/2023/04/11/national/diplomacy/Korea-United-States-eavesdropping/20230411172614768.html", 
        "https://en.yna.co.kr/view/AEN20230410005251315",  
        "https://www.voanews.com/a/ahead-of-biden-yoon-meeting-us-accused-of-spying-on-south-korea/7043685.html",
        "https://www.bbc.com/news/world-asia-19207086",
        "https://apjjf.org/-Lee-Sang-tae/1728/article.html"]
    
    for url in page_url_list:
        print(url)
        sleep(2)
    
    return page_url_list

# page_url_list = search(proposition)


# ==============================
# SUMY
# ==============================


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

    print(len(article_list), "articles collected from web sites.")
    
    return article_list

# article_list = sumy(page_url_list)

# ==============================
# Summarize each Article
# ==============================


from transformers import pipeline
from transformers import AutoTokenizer

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def count_token(article):
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
    tokenized_input = tokenizer(article, truncation=True, padding=True)
    return len(tokenized_input['input_ids'])

def summarize(article_list):
    summary_list = []
    
    for i, article in enumerate(article_list):
        print(i + 1)
        num_token = count_token(article)
        summ = summarizer(article, max_length=min(num_token // 5, 300), min_length=num_token // 10, do_sample=False)
        summary_list.append(summ[0]['summary_text'])
        print(summ[0]['summary_text'])

    print("num:", len(summary_list), "article summarized.")
    
    return summary_list

# summary_list = summarize(article_list)


# ==============================
# Evaluate Sentence Similarity
# ==============================

def split_to_sentences(summary_list):
    for i, summary in enumerate(summary_list):
        print(i)
        for sentence in summary.split(". "):
            print("- " + sentence)
        print("-"*20, "\n\n")

from sentence_transformers import SentenceTransformer, util
import scipy


def evaluate(proposition, negation, page_url_list, summary_list):

    threshold = 0.7

    summarizer = dict()
    summarizer['model'] = SentenceTransformer('roberta-large-nli-mean-tokens')
    
    similarity_list = []

    v_proposition = summarizer['model'].encode(proposition)
    v_negation = summarizer['model'].encode(negation)

    for i, summary in enumerate(summary_list):
        b = {'index':i, 'sentences':[], 'page_url':page_url_list[i]}

        value_list = []

        for sentence in summary.split(". "):
            d = {}

            v_sentence = summarizer['model'].encode(sentence)
            pro_cos_sim = float(util.pytorch_cos_sim(v_proposition, v_sentence))
            con_cos_sim = -float(util.pytorch_cos_sim(v_negation, v_sentence))

            # pros
            if pro_cos_sim > -con_cos_sim and pro_cos_sim >= threshold:
                flag = "pros"
                value_list.append(pro_cos_sim)

            # cons
            elif pro_cos_sim < -con_cos_sim and -con_cos_sim >= threshold:
                flag = "cons"
                value_list.append(con_cos_sim)

            else:
                flag = "neut"

            d['sentence'] = sentence
            d['flag'] = flag
            d['pro_cos_sim'] = pro_cos_sim
            d['con_cos_sim'] = con_cos_sim

            b['sentences'].append(d)

        if len(value_list) > 0:
            similarity = sum(value_list) / len(value_list)
            b['similarity'] = similarity
        else:
            b['similarity'] = 0.

        similarity_list.append(b)
        
    return similarity_list

# similarity_list = evaluate(proposition, negation, page_url_list, summary_list)


# ==============================
# Split each sentence to pros and cons
# ==============================


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

# pros_list, cons_list = split_pros_cons(similarity_list)

