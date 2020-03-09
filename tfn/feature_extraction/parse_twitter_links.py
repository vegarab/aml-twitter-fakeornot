# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:56:51 2020

@author: aghou
"""

import pandas as pd
import os
import re
import requests
import time
from bs4 import BeautifulSoup

'''
The goal of this module is to return a dataframe containing metadata for the webpages
referred to in the tweets

Output dataframe contains columns:
    "train_index", "url", "redirect_url", 
    "title", "status_code", "error_log"
'''

OUTPUT_PATH = 'tfn/data/url.csv'

def _get_twitter_urls(text):
    # Returns URLs in the tweet text
    pattern = 'https?://t.co/.{10}'
    twitter_urls = re.findall(pattern, text)
    return twitter_urls

def _get_redirect_info(row):
    # Returns the following:
    #   - Index num of corresponding tweet
    #   - Twitter url (t.co format)
    #   - URL after redirect
    #   - Title of redirect page
    #   - Status code of response
    #   - Error desc (if any)

    # Folder where backups of the requested html files will be stored
    HTML_STORE_PATH = 'tfn/html-store'

    train_index = row["train_index"]
    url = row["twitter_url"]
    identifier = url[-10:]
    
    backup_file_name = str(train_index) + '-' + str(identifier) + '.html'
    
    try:
        req = requests.get(url, timeout=10)
    except ConnectionError as e:
        time.sleep(10)
        return train_index, url, None, None, None, None
    except Exception as e:
        return train_index, url, None, None, None, None
    
    print(train_index, req.url)
    try:
        html_content = req.text
        
        # Save backup file of html for redirects
        with open(os.path.join(HTML_STORE_PATH, backup_file_name), 
                  'w+', 
                  encoding="utf-8"
            ) as f:
            f.write(html_content)

        soup = BeautifulSoup(req.text, 'lxml')
        
        return train_index, url, req.url, soup.title.text, req.status_code, None
    
    except Exception as e:
        print(e)
        return train_index, url, req.url, None, req.status_code, e

def extract_twitter_url_info():
    DATA_PATH = 'tfn/data/train.csv'
    
    train_df = pd.read_csv(DATA_PATH)
    
    url_df = pd.DataFrame()
    url_df['twitter_urls'] = train_df['text'].apply(_get_twitter_urls)
    
    # Each url is plit into individual rows
    url_df_exploded = url_df\
        .explode('twitter_urls')\
        .reset_index()\
        .rename(columns={'index': 'train_index',
                         'twitter_urls': 'twitter_url'})
    
    url_df_exploded = url_df_exploded[url_df_exploded['twitter_url'].notna()]
    
    url_data_columns = ["train_index", "url", "redirect_url", 
                        "title", "status_code", "error_log"]
    
    url_data = url_df_exploded.apply(_get_redirect_info, 1, result_type="expand")
    url_data.columns = url_data_columns
    
    return url_data
    
if __name__ == '__main__':
    url_data = extract_twitter_url_info()
    url_data.to_csv(OUTPUT_PATH)
