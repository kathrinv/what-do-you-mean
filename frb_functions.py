# library imports
    # webscraping
from selenium import webdriver
import re
import time
import numpy as np
import pandas as pd

#natural language processing
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet, stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer 

from tqdm import tqdm_notebook as tqdm
import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel
import pyLDAvis.gensim

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


def navigate_frb_speeches():
    """
    Navigates the Fed Speeches website 
    and calls get_frb_article_links helper
    function to scrape the urls to all Fed 
    speeches from the Fed website (non-archived
    speeches up until 2006). 
    
    Returns:
    list: Speech urls for all non-archived
    speeches on the Feb website.
    """
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    browser.get("https://www.federalreserve.gov/newsevents/speeches.htm")
    article_urls = []
    new_urls = get_frb_article_links(browser)
    while not article_urls or article_urls[-1] != new_urls[-1]:
        article_urls += get_frb_article_links(browser)
        next_button = browser.find_element_by_css_selector("a[ng-click='selectPage(page + 1, $event)']")
        next_button.click()
        new_urls = get_frb_article_links(browser)
        time.sleep(np.random.randint(5,10))
    browser.close()
    return article_urls
    
def get_frb_article_links(browser):
    """
    Helper function for navigagte_frb_speeches.
    (only works for non-archived speeches)
    
    Returns:
    list: Speech urls for the current
    page of speeches.
    """
    new_urls = []
    articles = browser.find_elements_by_class_name('itemTitle')
    for article in articles:
        url = article.find_element_by_tag_name('a').get_attribute('href')
        new_urls.append(url)
    return new_urls

def get_frb_speech_text(url_lst):
    """
    Accesses and scrapes all the speech text from a
    list of urls provided. Only works for non-archived
    speeches on the Fed website.
    
    Parameters: 
    url_lst (list): list of speech urls to scrape
    
    Returns:
    list: A list of lists that contains
        the speech url, date, title, speaker, location, 
        and complete text for all speeches in the 
        url_lst.
    """
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    frb_articles = []
    for url in url_lst:
        article_details = []
        article_details.append(url)
        browser.get(url)
        article_times = browser.find_elements_by_class_name('article__time')
        article_details.append(article_times[0].text)
        article_titles = browser.find_elements_by_class_name('title')
        article_details.append(article_titles[0].text)
        article_speakers = browser.find_elements_by_class_name('speaker')
        article_details.append(article_speakers[0].text)
        article_locations = browser.find_elements_by_class_name('location')
        article_details.append(article_locations[0].text)
        article_texts = browser.find_elements_by_xpath('//*[@id="article"]/div[3]')
        article_details.append(article_texts[0].text)
        frb_articles.append(article_details)
        time.sleep(np.random.randint(5,10))
    browser.close()
    return frb_articles

def get_frb_article_links_archived(browser):
    new_urls = []
    new_titles = []
    new_speakers = []
    new_locations = []
    new_dates = []
    speeches = browser.find_element_by_id('speechIndex')
    speech_urls = speeches.find_elements_by_tag_name('a')
    for speech in speech_urls:
        url = speech.get_attribute('href')
        new_urls.append(url)
        title = speech.text
        new_titles.append(title)
    speech_dates = speeches.find_elements_by_tag_name('li')
    for speech in speech_dates:
        date_ = re.findall(r'(?<=)(\S+ \d+, \d{4})', speech.text)[0]
        new_dates.append(date_)
    speech_speakers = speeches.find_elements_by_class_name('speaker')
    for speaker in speech_speakers:
        new_speakers.append(speaker.text)
    speech_locations = speeches.find_elements_by_class_name('location')
    for location in speech_locations:
        new_locations.append(location.text)
    return new_urls, new_titles, new_speakers, new_locations, new_dates

def navigate_frb_archived_speeches():
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    browser.get("https://www.federalreserve.gov/newsevents/speech/speeches-archive.htm")
    speech_urls = []
    speakers = []
    locations = []
    dates_ = []
    titles = []
    year_links = []

    list_of_years = browser.find_element_by_xpath('//*[@id="article"]/div/div/div/ul')
    all_year_links = list_of_years.find_elements_by_tag_name("a")
    for year_link in all_year_links:
        url = year_link.get_attribute('href')
        year_links.append(url)
    for url in year_links:
        browser.get(url)
        new_urls, new_titles, new_speakers, new_locations, new_dates = get_frb_article_links_archived(browser)
        speech_urls = speech_urls + new_urls
        titles = titles + new_titles
        speakers = speakers + new_speakers
        locations = locations + new_locations
        dates_ = dates_ + new_dates
        time.sleep(np.random.randint(5,10))
    browser.close()
    # removing extra url accidentally picked up
    del titles[-118]
    del speech_urls[-118]
    return speech_urls, speakers, locations, dates_, titles

def get_frb_speech_text_archived(url_lst):
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    speech_text = []
    for url in url_lst:
        browser.get(url)
        paragraphs = browser.find_elements_by_tag_name('p')
        complete_text = ""
        for paragraph in paragraphs:
            complete_text += ' ' + paragraph.text
        speech_text.append(complete_text)
        time.sleep(np.random.randint(5,10))
    browser.close()
    return speech_text

def navigate_fomc_speeches():
    """
    """
    fomc_urls = []
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    browser.get("https://www.federalreserve.gov/newsevents/pressreleases.htm")
    new_urls = get_fomc_article_links(browser)
    while not fomc_urls or (not new_urls or fomc_urls[-1] != new_urls[-1]):
        fomc_urls += get_fomc_article_links(browser)
        time.sleep(np.random.randint(5,10))
        next_button = browser.find_element_by_css_selector("a[ng-click='selectPage(page + 1, $event)']")
        next_button.click()
        new_urls = get_fomc_article_links(browser)
    browser.close()
    return fomc_urls

def get_fomc_article_links(browser):
    """
    """
    new_urls = []
    speeches = browser.find_elements_by_class_name('itemTitle')
    for speech in speeches:
        if re.findall(r'FOMC statement', speech.text):
            new_urls.append(speech.find_element_by_tag_name('a').get_attribute('href'))
    return new_urls

def get_fomc_speech_text(url_lst):
    """

    """
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    fomc_speeches = []
    for url in url_lst:
        article_details = []
        article_details.append(url)
        browser.get(url)
        article_times = browser.find_elements_by_class_name('article__time')
        article_details.append(article_times[0].text)
        article_titles = browser.find_elements_by_class_name('title')
        article_details.append(article_titles[0].text)
        article_texts = browser.find_elements_by_xpath('//*[@id="article"]/div[3]')
        article_details.append(article_texts[0].text)
        fomc_speeches.append(article_details)
        time.sleep(np.random.randint(5,10))
    browser.close()
    return fomc_speeches


def navigate_fomc_archived_speeches():
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    browser.get("https://www.federalreserve.gov/newsevents/pressreleases/press-release-archive.htm")
    fomc_urls = []
    titles = []
    year_links = []

    list_of_years = browser.find_element_by_xpath('//*[@id="article"]/div/div/div/ul')
    all_year_links = list_of_years.find_elements_by_tag_name("a")
    for year_link in all_year_links:
        url = year_link.get_attribute('href')
        year_links.append(url)
    for url in year_links:
        browser.get(url)
        new_urls, new_titles = get_fomc_links_archived(browser)
        fomc_urls = fomc_urls + new_urls
        titles = titles + new_titles
        time.sleep(np.random.randint(5,10))
    browser.close()
    return fomc_urls, titles

def get_fomc_links_archived(browser):
    new_urls = []
    new_titles = []
    releases = browser.find_element_by_id('releaseIndex')
    release_urls = releases.find_elements_by_tag_name('a')
    for release in release_urls:
        if re.findall(r'FOMC [Ss]tatement', release.text):
            url = release.get_attribute('href')
            new_urls.append(url)
            title = release.text
            new_titles.append(title)
    return new_urls, new_titles

def get_fomc_text_archived(url_lst):
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    speech_text = []
    fomc_dates = []
    for url in url_lst:
        browser.get(url)
        paragraphs = browser.find_elements_by_tag_name('p')
        complete_text = ""
        for paragraph in paragraphs:
            complete_text += ' ' + paragraph.text
        speech_text.append(complete_text)
        date_ = browser.find_elements_by_tag_name('i')[0]
        date_ = re.findall(r'(?<=[rR]elease [dD]ate: )(\w* \d*,? \d*)', date_.text)[0]
        fomc_dates.append(date_)
        time.sleep(np.random.randint(5,10))
    browser.close()
    return speech_text, fomc_dates

def get_fed_funds_rates(archived=False):
    # initiating selenium Chrome webdriver instance
    option = webdriver.ChromeOptions()
    option.add_argument(" — incognito")
    browser = webdriver.Chrome(options=option)
    if not archived:
        browser.get('https://www.federalreserve.gov/monetarypolicy/openmarket.htm')
    else:
        browser.get('https://www.federalreserve.gov/monetarypolicy/openmarket_archive.htm')
    
    years_txt = []
    years = browser.find_elements_by_tag_name('h4')
    if not archived:
        years = years[1:]
    for year in years:
        years_txt.append(year.text)
    
    dates_ = []
    inc = []
    dec = []
    target = []
    
    rate_tables = browser.find_elements_by_class_name('data-table')
    for i, table in enumerate(rate_tables):
        for j, td in enumerate(table.find_elements_by_tag_name('td')):
            if (j+1) % 4 == 1:
                dates_.append(td.text + ", " + years_txt[i])
            elif (j+1) % 4 == 2:
                inc.append(td.text)
            elif (j+1) % 4 == 3:
                dec.append(td.text)
            elif (j+1) % 4 == 0:
                target.append(td.text)
    browser.close()
    return dates_, inc, dec, target



# Text cleaning
def remove_references(text):
    references_loc = text.find('\nReferences\n')
    if references_loc != -1:
        text = text[:references_loc]
    return_to_text_loc = text.find('[Rr]eturn to text\n')
    if return_to_text_loc != -1:
        text = text[:return_to_text_loc]
    concluding_remarks_loc = text.find('These remarks represent my own views, which do not necessarily represent those of the Federal Reserve Board or the Federal Open Market Committee.')
    if concluding_remarks_loc != -1:
        text = text[:concluding_remarks_loc]
    return text

def clean_speech_text(df):
    df_new = df.copy()
    df_new.loc['full_text'] = df_new['full_text'].apply(lambda x: remove_references(x))
    df_new.loc['full_text'] = df_new['full_text'].str.replace('\n', ' ')
    df_new.loc['full_text'] = df_new['full_text'].apply(lambda x: re.sub(r'(http)\S+(htm)(l)?', '', x))
    df_new.loc['full_text'] = df_new['full_text'].apply(lambda x: re.sub(r'(www.)\S+', '', x))
    df_new.loc['full_text'] = df_new['full_text'].apply(lambda x: re.sub(r'[\d]', '', x))
    df_new.loc['full_text'] = df_new['full_text'].str.replace('—', ' ')
    df_new.loc['full_text'] = df_new['full_text'].str.replace('-', ' ')
    df_new.loc['full_text'] = df_new['full_text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df_new.loc['full_text'] = df_new['full_text'].apply(lambda x: re.sub(r'([Rr]eturn to text)', '', x))
    df_new.loc['full_text'] = df_new['full_text'].apply(lambda x: re.sub(r'([Pp]lay [vV]ideo)', '', x))
    return df_new


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_speech_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens_lower = [w.lower() for w in nltk.word_tokenize(text)]
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens_lower]

def count_unique_words(text):
    return len(set(text))

def tokenize_and_remove_stopwords(text):
    tokens = word_tokenize(text)
    stopped_tokens = [w for w in tokens if w not in stopwords_without_punct]
    return stopped_tokens

def get_most_common_words(tokens, num=20):
    fdist = FreqDist(tokens)
    return fdist.most_common(num)

def remove_stop_words(tokens_list):
    stopwords_without_punct = []
    for word in stopwords.words('english'):
        word = word.replace("'", "")
        stopwords_without_punct.append(word)
    stopped_tokens = [w for w in tokens_list if w not in stopwords_without_punct]
    return [w for w in stopped_tokens if len(w) > 2]

def convert_to_datetime(df):
    df_new = df.copy()
    df_new.loc['speech_datetime'] = df_new['speech_date'].apply(lambda x: pd.to_datetime(x))
    df_new.loc['speech_year'] = df_new['speech_datetime'].apply(lambda x: x.year)
    df_new.loc['speech_month'] = df_new['speech_datetime'].apply(lambda x: x.month)
    return df_new

def plot_most_common_words(df, article_num=9):
    fig = plt.figure(figsize=(15, 6))
    fig.suptitle(f"Most common words in Speech: {df.iloc[article_num]['title']}")
    left = fig.add_subplot(121)
    right = fig.add_subplot(122)
    
    # left subplot without stop words
    sns.barplot(x=[x[0] for x in df.iloc[article_num]['common_20_stopped_lemm_words']],
            y=[x[1] for x in df.iloc[article_num]['common_20_stopped_lemm_words']], ax=left, color='#ffd966')#palette = mycmap)
    left.set_xticklabels(left.get_xticklabels(), rotation=45, horizontalalignment="right")
    left.set_title('Lemmatized Tokens with Stop Words Removed')
    
    # right subplot with all tokens
    sns.barplot(x=[x[0] for x in df.iloc[article_num]['common_20_lemm_words']],
            y=[x[1] for x in df.iloc[article_num]['common_20_lemm_words']], ax=right, color='gray')#palette = mycmap)
    right.set_xticklabels(right.get_xticklabels(), rotation=45, horizontalalignment="right")      
    right.set_title('Lemmatized Tokens')
                 
    plt.show()
    




