import pandas as pd
import os
import nltk
nltk.download('punkt')


def get_uploaded_example(decoded, langid):
    sentences = nltk.sent_tokenize(decoded, language=langid)
    return sentences


def get_generic_translations():
    with open('data/multiple-sentences-translations.txt') as f:
        sentences = [line.strip() for line in f]
        return sentences


def get_candas_doc(candas_keyword):
    df = pd.read_table(f'./data/candas/{candas_keyword}.tsv')
    text = df['Text'].get(0)
    sentences = nltk.sent_tokenize(text, "slovene")
    return sentences


def get_candas_doc_metadata(candas_keyword):
    df = pd.read_table(f'./data/candas_metadata/{candas_keyword}_sentences.tsv')
    sentences = df['sentence'].tolist()
    metadata = []
    for source, sentiment, year, bias in zip(df['source'], df['sentiment'], df['year_published'], df['bias']):
        meta = "<br>".join([f'Source: {source}', f'Sentiment: {sentiment}', f'Year: {year}', f'Bias: {bias}'])
        metadata.append(meta)
    return sentences, metadata