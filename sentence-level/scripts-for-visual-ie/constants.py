import datetime
from dateutil import parser
import re
import string
import json

vocab = ' ' + string.ascii_letters + string.digits + string.punctuation

character2id = {c:idx for idx, c in enumerate(vocab)}
UNK_CHAR = len(character2id)
CHAR_VOCAB_SIZE = len(vocab) + 1

def get_char_id(c):
    return character2id[c] if c in character2id else UNK_CHAR

WORD_VOCAB_SIZE = 10000
sorted_word = json.load(open('visual_vocab.json'))
words = [t[0] for t in sorted_word[:WORD_VOCAB_SIZE]]

PAD_WORD = 0
UNK_WORD = 1
word2id = {'PAD': PAD_WORD, 'UNK': UNK_WORD}
for w in words:
    word2id[w] = len(word2id)
    if len(word2id) == WORD_VOCAB_SIZE:
        break

def get_word_id(w):
    w = w.lower()
    return word2id[w] if w in word2id else UNK_WORD


MAX_WORD_LEN = 20
MAX_SENT_LEN = 60
MAX_DOCU_LEN = 200

# preprocesssing

STR_MATCH = 0
EXACT_MATCH = 1
DATE_MATCH = 2

attrs = {
    # "Initial Receipt Date": DATE_MATCH, 
    "Patient Initials": EXACT_MATCH, 
    "Patient Age": EXACT_MATCH, 
    "Patient Date of Birth": DATE_MATCH, 
    "Product (as reported)": EXACT_MATCH, 
    "Reported Event": EXACT_MATCH,
    "Reporter First Name": EXACT_MATCH, 
    "Reporter Last Name": EXACT_MATCH, 
    "Reporter City": EXACT_MATCH
}

attr2id = {'O': 0}
id2attr = {0: 'O'}
for attr in attrs:
    attr2id[attr] = len(attr2id)
    id2attr[attr2id[attr]] = attr

# tag2id = {'O': 0}
# id2tag = {0: 'O'}

def B_tag(attr):
    return 'B-'+attr 

def I_tag(attr):
    return 'I-'+attr

# for attr in attrs:
#     tag = B_tag(attr)
#     tag2id[tag] = len(tag2id)
#     id2tag[tag2id[tag]] = tag
#     tag = I_tag(attr)
#     tag2id[tag] = len(tag2id)
#     id2tag[tag2id[tag]] = tag

tag2id = {"B-Reporter Last Name": 1, "I-Product (as reported)": 4, "B-Patient Initials": 11, "I-Reporter First Name": 16, "I-Reporter City": 14, "B-Reporter City": 13, "I-Reporter Last Name": 2, "B-Product (as reported)": 3, "O": 0, "B-Patient Date of Birth": 7, "B-Reported Event": 9, "I-Reported Event": 10, "I-Patient Date of Birth": 8, "B-Reporter First Name": 15, "I-Patient Initials": 12, "I-Patient Age": 6, "B-Patient Age": 5}
id2tag = {_id:tag for tag,_id in tag2id.items()}


def get_attr_from_tag_id(tag_id):
    tag = id2tag[tag_id]
    return tag[2:]


ignore_attrs = [] # ["Initial Receipt Date"]
ignore_attrs_id = [] # [attr2id[attr] for attr in ignore_attrs]

# postprocessing

SINGLE_WORD = 0
SINGLE_SENT = 1
SINGLE_DATE = 2
MULTIP_WORD = 3
MULTIP_SENT = 4
attr2anstype = {
    # "Initial Receipt Date": SINGLE_DATE, 
    "Patient Initials": SINGLE_WORD, 
    "Patient Age": SINGLE_WORD,
    "Patient Date of Birth": SINGLE_DATE, 
    "Product (as reported)": MULTIP_SENT, 
    "Reported Event": MULTIP_SENT,
    "Reporter First Name": MULTIP_WORD, 
    "Reporter Last Name": MULTIP_WORD,
    "Reporter City": MULTIP_SENT
}


def isdate(s):
    try:
        t = parser.parse(s)
        return t.year >= 1900
    except:
        return False

def samedate(s1, s2):
    try:
        d1 = parser.parse(s1)
        d2 = parser.parse(s2, dayfirst=False)
        d3 = parser.parse(s2, dayfirst=True)
        return d1.date() == d2.date() or d1.date() == d3.date()
    except:
        return False


def process_annotation(attr, text):
    if isinstance(text, datetime.datetime):
        return [text.strftime("%Y-%m-%d")]
    if not isinstance(text, str):
        text = str(text)
    if attr == "Patient Age":
        text = text.replace(' Years', '')
    if attr == "Patient Initials":
        text = text.upper()
    invalid = ['', 'empty', '--', 'nan', '??', '???', '????']
    if text in invalid:
        return []
    splitting = re.compile(r"([0-9]\)(\s)+|\n+| : )")
    text = re.sub(splitting, "\n", text)
    substituting = re.compile(r"( 00:00:00| : --|)")
    text = re.sub(substituting, "", text)
    return [x for x in text.split("\n") if x.strip() not in invalid]
