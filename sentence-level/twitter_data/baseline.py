import json
import numpy as np

# task = 'education'
task = 'job'

f = open('%s_user.json'%task)
users = {}
for line in f:
    u = json.loads(line)
    users[u['id']] = u

n = len(users)
# train_size = int(n*0.6)
# print(train_size)
# users = list(users.values())
# train = users[:train_size]
# test = users[train_size:]

f = open('%s_split.json'%task)
split_list = json.load(f)

prec_list = []
recall_list = []
f1_list = []

for _id, obj in enumerate(split_list):
    train = [users[_] for _ in obj['train']]
    test = [users[_] for _ in obj['test']]
    print(_id, len(train), len(test))

    ecount = {}
    for u in train:
        for entity in u['pos_entity']:
            if entity not in ecount:
                ecount[entity] = 0
            ecount[entity] += 1

    # print(ecount)

    def eval(hit, pred_cnt, true_cnt):
        prec = hit / pred_cnt
        recall = hit / true_cnt
        f1 = 2*prec*recall / (prec+recall)
        print(prec, recall, f1)
        return prec, recall, f1

    hit, true_cnt, pred_cnt = 0,0,0
    error_cnt = 0
    for user in test:
        tweet = user['pos_tweet'] + user['neg_tweet']
        _hit, _true, _pred = 0,0,0
        _list = []
        for t in tweet:
            n = len(t['sent'])
            for i in range(n):
                for j in range(i+1,n+1):
                    if j-i>=5:
                        break
                    et = ' '.join(t['sent'][i:j])
                    if et in ecount and ecount[et]>=5:
                        if et in user['pos_entity']:
                            _hit += 1
                        _pred += 1
                    if et in user['pos_entity']:
                        _true += 1
                        _list.append(t)
        if _true != sum([len(tweet['match']) for tweet in user['pos_tweet']]):
            error_cnt += 1

        hit += _hit
        pred_cnt += _pred
        true_cnt += _true
    # print(hit, pred_cnt, true_cnt)
    p,r,f = eval(hit, pred_cnt, true_cnt)
    prec_list.append(p)
    recall_list.append(r)
    f1_list.append(f)
    # print(error_cnt)

print(np.mean(prec_list), np.mean(recall_list), np.mean(f1_list))

