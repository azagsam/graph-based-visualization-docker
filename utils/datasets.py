import pandas as pd
import os
import random

def get_uploaded_example(nlp, decoded):

    def get_sentences(text):
        doc = nlp(text)
        sentences = []
        for sent in doc.sentences:
            sentences.append(
                ' '.join([f'<br>{word.text}' if idx % 20 == 0 else word.text for idx, word in enumerate(sent.words)]))
        return sentences
    sentences = get_sentences(decoded)
    return sentences


def get_parlamint(row):
    df = pd.read_csv('/home/ales/Documents/Extended/datasets/ParlaMint/output/ParlaMint-SI.csv')
    df_filtered = df[df['SPEAKER_ROLE'] == 'regular']
    df_filtered.reset_index(inplace=True)

    g = df_filtered.groupby('SESSION_ID')

    for idx, (name, data) in enumerate(g):
        sentences = []
        if idx == row:
            for speech in data['TEXT_sentences']:
                # print('\n', speech)
                sents = speech.split('</s>')
                for s in sents:
                    s_with_break = " ".join([f'<br>{word}' if idx % 20 == 0 else word for idx, word in enumerate(s.split())])
                    sentences.append(s_with_break)
            return sentences


def get_candas(nlp, row):

    def get_sentences(text):
        doc = nlp(text)
        sentences = []
        for sent in doc.sentences:
            sentences.append(
                ' '.join([f'<br>{word.text}' if idx % 20 == 0 else word.text for idx, word in enumerate(sent.words)]))
        return sentences

    df = pd.DataFrame()
    for file in os.scandir('./data/docs'):
        f = pd.read_table(file.path)
        f['file'] = [file.name] * len(f)
        df = pd.concat([df, f])
    df.reset_index(inplace=True)

    text = df['Text'].get(row)
    sentences = get_sentences(text)
    return sentences

def get_kas(nlp, row):
    ...