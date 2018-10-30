import os
import sys
import json
import pandas as pd
from dateutil import parser
import datetime
import string
import re
import argparse
from multiprocessing import Pool
from nltk.tokenize.moses import MosesTokenizer
from constants import *

pser = argparse.ArgumentParser()

pser.add_argument('--case', type=str, default='tmp_selected_case.txt')
pser.add_argument('--excel', type=str, default='Cons_Full_training.xlsx')

args = pser.parse_args()

moses = MosesTokenizer()
v_overlap = 0.5
v_margin = 2.
h_overlap = 0.5

excel = pd.read_excel(args.excel)
record = {row.loc['Case Number']: row for idx, row in excel.iterrows()}


def tokenize(s):
	tmp = moses.tokenize(s, agressive_dash_splits=True)
	return [w if w != u'\\@-\\@' else '-' for w in tmp]

def date_match(src, annotations):
    res = []
    for tgt in annotations:
        if not isdate(tgt):
            continue
        date = parser.parse(tgt).date()
        if str(date.year) not in ' '.join(src):
            continue
        n = len(src)
        l = 0
        while l < n:
            for r in range(l, n):
                s1 = ' '.join(src[l:r+1])
                if samedate(s1, tgt):
                    res.append([l, r+1])
                    l = r
                    break
            l += 1
        # for idx, word in enumerate(src):
        #     if samedate(word, tgt):
        #         res.append([idx, idx+1])
    return res

def exact_match(src, annotations, attr):
    # debug=False
    # if ' '.join(src) =='KH':
    #     debug=True
    #     print(src, annotations)
    def preprocess(origin_seq):
    	seq, idx = [], []
    	for i, w in enumerate(origin_seq):
    		if w not in string.punctuation:
	    		seq.append(w)
	    		idx.append(i)
        if attr != "Patient Initials":
            seq = [w.lower() for w in seq]
        else:
            seq = [w.replace('.', '') for w in seq]
        return seq, idx
    res = []
    for tgt in annotations:
        (src_seq, src_idx), (tgt_seq, _) = preprocess(src), preprocess(tokenize(tgt))
        src_len, tgt_len = len(src_seq), len(tgt_seq)
        if tgt_len == 0:
        	continue
        for i in range(src_len - tgt_len + 1):
            if src_seq[i:i+tgt_len] == tgt_seq:
            	# print(src_idx, i, i+tgt_len-1)
                res.append([src_idx[i], src_idx[i+tgt_len-1]+1])
    return res


def gen_graph(case_path):
    try:
        f = open(case_path + 'parsed.json', 'r')
        parsed = json.load(f)
    except:
        parsed = {'id':case_path.split('/')[-2], 'docs':{}}
        for name in os.listdir(case_path):
            if '.pdf.json' in name:
                obj = json.load(open(case_path+name, 'r'))
                for doc_id, doc in obj['docs'].items():
                    parsed['docs'][doc_id] = doc
    result = []
    result_text = []
    record_row = record[parsed['id']]
    matched = {attr: False for attr in attrs}
    for doc_id in parsed['docs']:
        doc = parsed['docs'][doc_id]
        for page_id in doc['pages']:
            page = doc['pages'][page_id]
            width, height = page['size']
            N = len(page['text'])
            graph = {'id':{'case':parsed['id'], 'doc':doc_id, 'page':page_id}, 
                'sent':[], 'pos':[], 'tag':[], 'edge':{'v':[],'h':[]}}
            for word in page['text']:
                graph['sent'].append(tokenize(word[1]))
                x1, y1, x2, y2 = word[0]
                x1, x2 = x1/width, x2/width
                y1, y2 = 1-y2/height, 1-y1/height
                graph['pos'].append([x1, y1, x2, y2])
            
            for sent in graph['sent']:
                tag = [0 for w in sent]
                graph['tag'].append(tag)
            for attr, match_type in attrs.items():
                annotations = process_annotation(attr, record_row.loc[attr])
                if len(annotations) == 0:
                    matched[attr] = True
                    continue
                for u in range(N):
                    sent = graph['sent'][u]
                    tag = graph['tag'][u]
                    if match_type == DATE_MATCH:
                        match = date_match(sent, annotations)
                    elif match_type == EXACT_MATCH:
                        match = exact_match(sent, annotations, attr)
                    if len(match) > 0:
                        # print(sent, anno)
                        matched[attr] = True
                        for l, r in match:
                            tgid = tag2id[B_tag(attr)]
                            tag[l] = tgid
                            for j in range(l+1, r):
                                tag[j] = tag2id[I_tag(attr)]

            def overlap(x1, x2, y1, y2):
                res = x2-x1 + y2-y1 - (max(y2,x2)-min(y1,x1))
                return max(res, 0)
            def check_vertical_edge(pos1, pos2):
                a_x1, a_y1, a_x2, a_y2 = pos1
                b_x1, b_y1, b_x2, b_y2 = pos2
                if b_y1 > a_y1 and overlap(a_x1,a_x2,b_x1,b_x2)>min(a_x2-a_x1,b_x2-b_x1)*v_overlap:
                    d = b_y1 - a_y2
                    if d < max(a_y2-a_y1,b_y2-b_y1)*v_margin:
                        return d
                return None
            def check_horizontal_edge(pos1, pos2):
                a_x1, a_y1, a_x2, a_y2 = pos1
                b_x1, b_y1, b_x2, b_y2 = pos2
                if b_x1 > a_x1 and overlap(a_y1,a_y2,b_y1,b_y2)>min(a_y2-a_y1,b_y2-b_y1)*h_overlap:
                    return b_x1 - a_x2
                return None
            for u in range(N):
                v, min_d = None, 1
                for _v in range(N):
                    d = check_vertical_edge(graph['pos'][u], graph['pos'][_v])
                    if d is not None and d < min_d:
                        min_d, v = d, _v
                if v is not None:
                    graph['edge']['v'].append([u,v])
            result.append(graph)

            lines = [[u] for u in range(N)]
            father = [u for u in range(N)]
            def get_father(u):
                if father[u] == u:
                    return u
                return get_father(father[u])
            for u in range(N):
                v, min_d = None, 1
                for _v in range(N):
                    d = check_horizontal_edge(graph['pos'][u], graph['pos'][_v])
                    if d is not None and d < min_d:
                        min_d, v = d, _v
                if v is not None:
                    graph['edge']['h'].append([u,v])
                    fu, fv = get_father(u), get_father(v)
                    lines[fu] = lines[fu] + lines[fv]
                    lines[fv] = []
                    father[fv] = fu
            text_obj = {'id':{'case':parsed['id'], 'doc':doc_id, 'page':page_id}, 
                'sent':[], 'pos':[], 'tag':[], 'edge':{'v':[],'h':[]}}
            for line in lines:
                if len(line) > 0:
                    sent = [] 
                    for u in line:
                        sent += graph['sent'][u]
                    tag = []
                    for u in line:
                        tag += graph['tag'][u]
                    text_obj['sent'].append(sent)
                    text_obj['tag'].append(tag)
                    text_obj['pos'].append([0,0,0,0])
            result_text.append(text_obj)

    num_matched_attr = 0
    for attr in attrs:
        if not matched[attr]:
            print(parsed['id'], attr,record_row.loc[attr])
        else:
            num_matched_attr += 1
    num_page = len(result)
    num_sent = sum([len(graph['sent']) for graph in result])
    output = open(case_path + 'graph_bio.json', 'w')
    for graph in result:
        output.write(json.dumps(graph) + '\n')
    output = open(case_path + 'textline_bio.json', 'w')
    for text_obj in result_text:
        output.write(json.dumps(text_obj) + '\n')
    return num_page, num_sent, num_matched_attr


def main():
    # case_list = []
    # for direc in os.listdir(base_path):
    #     path = base_path + direc + '/'
    #     case_list.append(path)
    f = open(args.case)
    case_list = [line.strip() for line in f]
    f.close()

    pool = Pool(processes=20)
    # result = [gen_graph(case) for case in case_list]
    result = pool.map(gen_graph, case_list)
    sys.stdout.flush()
    # output = open(args.output, 'w')
    # for idx, res in enumerate(result):
    #     num_page, num_sent, num_matched_attr = res
    #     if num_page < 100 and num_sent > 0 and num_matched_attr == len(attrs):
    #         output.write(case_list[idx] + '\n')
    # log = open('log.txt', 'w')
    # log.write('num_case: %d\n' % len(case_list))
    # log.write('num_valid_case: %d\n' % sum([1 for a,b,c in result if a < 100 and b > 0]))
    # output = [parse_case(path) for path in case_list]


if __name__ == '__main__':
    main()