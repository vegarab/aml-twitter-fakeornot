import os

import pandas as pd

from google.cloud import translate_v2 as translate
from google.oauth2.service_account import Credentials

_DATA_PATH = 'tfn/data'

'''As far as I can tell, we will need to set up Google services credentials.'''
_GOOGLE_CREDS_PATH = "CREDENTIALS HERE"

credentials = Credentials.from_service_account_file(
    _GOOGLE_CREDS_PATH
)

translate_client = translate.Client(credentials=credentials)

def doubleTranslate(text, target_language):
    double_translated = translate_client.translate(
        translate_client.translate(
            original_tweet_text,
            target_language=target_language
        )['translatedText'],
        target_language='en'
    )

    return double_translated

def get_data_from_csv():
    csv_path = os.path.join(_DATA_PATH, 'train.csv')
    df = pd.read_csv(csv_path, header=0)

    return df

# Takes array of tweets and outputs total characters [Translate API has free char limit]
def total_tweet_chars(text_arr):
    return sum(len(i) for i in text_arr)

def yn(text):
    while True:
        i = input(text)
        if i.lower() == 'y':
            return True
        elif i.lower() == 'n':
            return False

def create_translated_data():
    # TODO: Function to create translated csv and place in 'translation_data' dir
    pass

if __name__ == '__main__':
    df = get_data_from_csv()

    if yn(f'# of characters: {total_tweet_chars(df.text)} \nContinue? (y/n)\n'):
        '''# Execute doubleTranslate'''
        # df.text.apply(doubleTranslate, 'iw')
        pass
    else:
        print('Aborted')
        pass

