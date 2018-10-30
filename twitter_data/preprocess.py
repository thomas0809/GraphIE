import sys
import argparse
import json
import random
import copy
from tqdm import tqdm

random.seed(1)

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='education')
parser.add_argument('--base', type=str, default='./')

args = parser.parse_args()

users = {}

def get_match(a, b):
	n = len(a)
	m = len(b)
	res = []
	for i in range(n):
		if a[i:i+m] == b:
			res.append((i,i+m-1))
	return res

neg_users = {}
unmatched = []

def process_tweet(l_name, l_entity, l_text, tag):
	a = l_name.strip().split('$')
	if tag == 1:
		uid, uname, ufullname, ulabel = a
	else:
		uid, uname, ufullname = a
		ulabel = ''
		if uid not in users:
			neg_users[uid] = True
			return
	entity = l_entity.strip()
	if uid not in users:
		obj = {'id':uid, 'name':uname, 'full_name': ufullname, 'label':ulabel, 'pos_entity':[], 'pos_tweet':[], 'neg_tweet':[]}
		users[uid] = obj
	else:
		obj = users[uid]
	
	if tag == 1:
		if entity not in obj['pos_entity']:
			obj['pos_entity'].append(entity)
		elist = obj['pos_tweet']
	else:	
	 	elist = obj['neg_tweet']
			
	a = l_text.strip().split(' ')
	# b = entity.split(' ')		
	# match = get_match(a, b)
	# if len(match) == 0:
	# 	if tag == 1:
	# 		print(l_text + l_entity)
	# 	unmatched.append(1)
	match = []
	
	elist.append({'sent':a, 'match':match})


f_name = open(args.base+'%s_name.txt'%args.task)
f_entity = open(args.base+'%s_entity.txt'%args.task)
f_text = open(args.base+'%s_text.txt'%args.task)

for l_name, l_entity, l_text in tqdm(zip(f_name, f_entity, f_text)):
	process_tweet(l_name, l_entity, l_text, 1)


f_name = open(args.base+'negative_%s_name.txt'%args.task)
f_entity = open(args.base+'negative_%s_entity.txt'%args.task)
f_text = open(args.base+'negative_%s_text.txt'%args.task)

for l_name, l_entity, l_text in tqdm(zip(f_name, f_entity, f_text)):
	process_tweet(l_name, l_entity, l_text, 0)


# match the positive entities in negative tweets
for u in users:
	tweets = users[u]['pos_tweet'] + users[u]['neg_tweet']
	pos_list = []
	neg_list = []
	for t in tweets:
		for entity in users[u]['pos_entity']:
			a = t['sent']
			b = entity.split(' ')
			match = get_match(a, b)
			if len(match) > 0:
				t['match'] += match
		if len(t['match']) > 0:
			pos_list.append(t)
		else:
			neg_list.append(t)
	users[u]['pos_tweet'] = pos_list
	users[u]['neg_tweet'] = neg_list


cnt_pos_tweet = 0
cnt_neg_tweet = 0

MAX_POS_TWEET = 20
MAX_NEG_TWEET = 100
MAX_TWEET = 100
# if args.task == 'job':
# 	MAX_TWEET = 150

N_SAMPLE = 1
for u in users:
	assert len(users[u]['pos_tweet'])+len(users[u]['neg_tweet'])>0

print('num_user', len(users)*N_SAMPLE)
print(len(neg_users))
print(len(unmatched))

output_u = open('%s_user.json'%args.task, 'w')
for uid, full_obj in users.items():
	for i in range(N_SAMPLE):
		obj = copy.copy(full_obj)
		obj['id'] = obj['id'] + '-%d'%i
		n_pos_tweet = len(obj['pos_tweet'])
		n_neg_tweet = len(obj['neg_tweet'])
		if n_pos_tweet + n_neg_tweet > MAX_TWEET:
			random.shuffle(obj['pos_tweet'])
			random.shuffle(obj['neg_tweet'])
			n = max(1, int(MAX_TWEET*float(n_pos_tweet)/(n_pos_tweet+n_neg_tweet)))
			m = MAX_TWEET-n if n+n_neg_tweet>MAX_TWEET else n_neg_tweet
			obj['pos_tweet'] = obj['pos_tweet'][:n]
			obj['neg_tweet'] = obj['neg_tweet'][:m]
		cnt_pos_tweet += len(obj['pos_tweet'])
		cnt_neg_tweet += len(obj['neg_tweet'])
		output_u.write(json.dumps(obj) + '\n')


print('num_pos_tweet', cnt_pos_tweet)
print('num_neg_tweet', cnt_neg_tweet)


graph = {uid: [] for uid in users}

cnt_edge = 0

f_network = open(args.base+'network_%s.txt'%args.task)
for line in f_network:
	u, v = line.strip().split(' ')
	if u not in graph or v not in graph:
		continue
	cnt_edge += 1
	graph[u].append(v)
	graph[v].append(u)

print('num_edge', cnt_edge*N_SAMPLE)

for uid in graph:
	graph[uid] = list(set(graph[uid]))
	if uid in graph[uid]:
		graph[uid].remove(uid)
		# print(uid, 'self-loop!')


output_g = open('%s_graph.json'%args.task, 'w')
for uid in users:
	# print(users[uid]['label'])
	# print([users[vid]['label'] for vid in graph[uid]])
	# print()
	obj = {'center': uid, 'node': [uid]+graph[uid], 'edge': []}
	u2idx = {u:idx for idx, u in enumerate(obj['node'])}
	for u in u2idx:
		for v in graph[u]:
			if v in u2idx:
				obj['edge'].append([u2idx[u], u2idx[v]])
	for i in range(N_SAMPLE):
		new_obj = copy.deepcopy(obj)
		new_obj['center'] += '-%d'%i
		for j in range(len(new_obj['node'])):
			new_obj['node'][j] += '-%d'%i
		output_g.write(json.dumps(new_obj) + '\n')



