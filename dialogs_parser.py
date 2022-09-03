import argparse
import pandas as pd
from natasha import NamesExtractor, MorphVocab

from yargy import Parser, rule, or_, and_, not_
from yargy.interpretation import fact
from yargy.predicates import gram, eq, is_capitalized, length_eq 
from yargy.pipelines import morph_pipeline
from slovnet import NER
from navec import Navec

GREETINGS = ['здравствуйте', 'доброе утро', 'добрый день', 'добрый вечер']
FAREWELLS = ['до свидания', 'всего хорошего', 'всего доброго', 'хорошего дня', 'хорошего вечера']

parser = argparse.ArgumentParser()
parser.add_argument("indir", type=str, help='Input path to data')
parser.add_argument("outdir", type=str, help='Input path to save')
parser.add_argument("path_to_natasha", type=str, help='Input path to natasha. Example:F:\\Data_Science\\Anaconda\\Lib\\site-packages\\natasha')

args = parser.parse_args()

data_path = args.indir
path_to_save = args.outdir
path_to_natasha = args.path_to_natasha

navec = Navec.load(path_to_natasha + '\\data\\emb\\navec_news_v1_1B_250K_300d_100q.tar')
ner = NER.load(path_to_natasha + '\\data\\model\\slovnet_ner_news_v1.tar')
ner.navec(navec)

PT = eq('.')
TR = eq('-')

ORGFORM = morph_pipeline(['компания ', 'ооо ', 'оао ', 'зао ', 'ип ', 'индивидуальный предприниматель'])
ABBR = gram('Abbr')
SURN = gram('Surn')
NAME = gram('Name')
PATR = and_(gram('Patr'),not_(ABBR))

INIT = and_(length_eq(1), is_capitalized())
FRST_INIT = INIT
LST_INIT = INIT
ORGNAME = and_(gram('NOUN'))

ORGANIZATION = or_(
    rule(ORGFORM, SURN, FRST_INIT, PT, LST_INIT, PT),
    rule(ORGFORM, FRST_INIT, PT, LST_INIT, PT, SURN),
    rule(ORGFORM, ORGNAME, TR, ABBR),
    rule(ORGFORM, SURN, FRST_INIT, PT),
    rule(ORGFORM, FRST_INIT, PT, SURN),
    rule(ORGFORM, NAME, SURN, PATR),
    rule(ORGFORM, SURN, NAME, PATR),
    rule(ORGFORM, SURN, FRST_INIT),
    rule(ORGFORM, FRST_INIT, SURN),
    rule(ORGFORM, NAME, SURN),
    rule(ORGFORM, SURN, NAME),
    rule(ORGFORM, ORGNAME),
    rule(ORGFORM, ORGNAME, ORGNAME),
)

ORG = Parser(ORGANIZATION)

morph_vocab = MorphVocab()
extractor = NamesExtractor(morph_vocab)

def orgs_extract(text, parser):
    found_values = []
    for match in parser.findall(text):
        found_values = [token.value for token in match.tokens]
    return found_values

def extract_name(text):
    matches = extractor(text)
    ners = [_.fact.as_json for _ in matches]
    name = ''
    for ner in ners:
        if 'first' in ner.keys():
            name = ner['first']
    return name

data = pd.read_csv(data_path + '\\test_data.csv')
data['text'] = data['text'].str.lower()
data['greeting_phrase'] = data.apply(lambda x: x['text'] if True in [_ in x['text'] for _ in GREETINGS] 
                                                                and x['role'] == 'manager'
                                                             else '', 
                                                             axis=1)

data['manager_name'] = data.apply(lambda x: extract_name(x['text']) 
                                            if x['role'] == 'manager' and x['line_n'] <= 5 and 
                                                                           ('зовут' in x['text'] 
                                                                            or 'это' in  x['text']
                                                                            or 'вас беспокоит' in x['text'])
                                            else '', 
                                            axis=1)

data['presentation_phrase'] = data.apply(lambda x: x['text'] 
                                            if x['manager_name'] != '' 
                                            else '', 
                                            axis=1)

data['company'] = data.apply(lambda x: ' '.join(orgs_extract(x['text'], ORG)[1:]) 
                                            if x['role'] == 'manager' and x['line_n'] <= 5 
                                            else '', 
                                            axis=1)

data['farewell_phrase'] = data.apply(lambda x: x['text'] if True in [_ in x['text'] for _ in FAREWELLS] 
                                                                and x['role'] == 'manager'
                                                             else '', 
                                                             axis=1)


summary_dialogs = data.groupby(by='dlg_id').agg({'greeting_phrase' : 'max',
                                                 'presentation_phrase' : 'max',
                                                 'farewell_phrase' : 'max',
                                                 'manager_name' : 'max',
                                                 'company': 'max'}).reset_index()

summary_dialogs['is_greeting'] = summary_dialogs['greeting_phrase'] != ''
summary_dialogs['is_farewell'] = summary_dialogs['farewell_phrase'] != ''
summary_dialogs['correctness'] = summary_dialogs.apply(lambda x: x['is_greeting'] and x['is_farewell'], axis=1)


data.to_csv(path_to_save + '\\test_data_with_results.csv', index=0, encoding="utf-8-sig")
summary_dialogs.to_csv(path_to_save + '\\summary_results.csv', index=0, encoding="utf-8-sig")
