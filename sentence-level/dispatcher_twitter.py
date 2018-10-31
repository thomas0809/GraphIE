import argparse
import subprocess
import os
import multiprocessing
import datetime
import pickle
import json
import csv
import time
from smtplib import SMTP
from email import encoders
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import mimetypes
from email import message
from termcolor import colored
import itertools as it
import random


parser = argparse.ArgumentParser(description='Grid Search Dispatcher')
parser.add_argument("--num_gpu", nargs="?", type=int, default=3, help="num gpus available to process. Default assume all gpus < num are available.")
parser.add_argument('--task', type=str, default='education')
parser.add_argument("--log_dir", nargs="?", type=str, default="logs/", help="Directory for the log file. [Default: logs/].")
parser.add_argument('--result_path', nargs='?', type=str, default="grid_search.csv", help="path to store grid_search table")
parser.add_argument('--rerun_experiments', default=False, action='store_true', help='whether to rerun experiments with the same result file location')


args = parser.parse_args()
print(args)

args.log_dir = args.task + '_' + args.log_dir
if not os.path.isdir(args.log_dir): 
    os.mkdir(args.log_dir)

if __name__ == "__main__":
    model_args = {
            'model'         : ['lstm-lstm', 'lstm-gcn-lstm'],
            # 'filter_sizes'  : ['2,3,4', '2,3,4,5'],
            # 'n_filter'      : [64],
            # 'd_embed'       : [64, 128],
            'd_graph'       : ['64'],
            # 'd_pos_embed'   : [16, 32],
            # 'dropout'       : [0, 0.1, 0.2, 0.4],
            # 'final'         : ['linear', 'attn', 'lstm'],
            'lr'            : [1e-3],
            # 'wd'            : [0],
            # 'entity_classification': [False, True],
            'weight_balance': [1],
            'crf'           : [True, False],
            'task'          : [args.task]
            }

    model_keys = sorted(model_args)

    gpu_queues = {}
    for q in range(args.num_gpu):
        gpu_queues[q] = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    results = []

    def launch_experiment(gpu, cur_args):

        log_name = ""
        for k in model_keys:
            if k != 'model' and len(model_args[k]) <= 1:
                continue
            log_name += "{}_{}_".format(k, cur_args[k])
        # print(log_name)
        log_name = log_name[:-1]
        log_stem = os.path.join(args.log_dir, log_name)
        log_path = log_stem + '.log'
        res_path = log_stem + '_results.json'

        experiment_string = "CUDA_VISIBLE_DEVICES={} python train_twitter.py --save_path={} ".format(gpu%args.num_gpu, log_stem)
        # 012, 001, 223, 220
        for k in model_keys:
            if type(cur_args[k]) == type(True):
                if cur_args[k] == True:
                    experiment_string += "--{} ".format(k)
            else: 
                experiment_string += "--{}='{}' ".format(k, cur_args[k])
        # print(experiment_string)
        
        ##forward logs to logfile
        shell_cmd = "{} > {} 2>&1".format(experiment_string, log_path)
        print("Time {}, launched exp: {}".format(str(datetime.datetime.now()), shell_cmd))
        if not os.path.exists(res_path) or args.rerun_experiments:
            subprocess.call(shell_cmd, shell=True)

        # if not os.path.exists(res_path):
        #     # running this process failed, alert me
        #     send_email("Dispatcher, Alert!", 
        #             "ALERT! job:[{}] has crashed! Check logfile at:[{}]".format(experiment_string, log_path))

        return res_path

    def worker(gpu, queue, done_queue):
        while not queue.empty():
            cur_args = queue.get()
            if cur_args is None:
                return
            done_queue.put(launch_experiment(gpu, cur_args))

    indx = 0
    num_jobs = 0
    all_args = it.product(*(model_args[k] for k in model_keys))
    all_args_dict = [dict(zip(model_keys, arg)) for arg in all_args]

    for cur_args in all_args_dict:
        gpu_queues[indx].put(cur_args)
        indx = (indx + 1) % args.num_gpu
        num_jobs += 1

    for gpu in range(args.num_gpu):
        job_queue = gpu_queues[gpu]
        print("Start gpu worker {} with {} jobs".format(gpu, job_queue.qsize()))
        multiprocessing.Process(target=worker, args=(gpu, job_queue, done_queue)).start()
    print() 

    keys_to_display = [
        #'train_loss','train_acc','train_recall','train_prec','train_f1',
        #'valid_loss','valid_acc',
        'valid_recall','valid_prec','valid_f1',
        #'test_loss','test_acc',
        'test_recall','test_prec','test_f1',
        'entity_prec', 'entity_recall', 'entity_f1'
        ] + model_keys

    for _ in range(num_jobs):
        result_path = done_queue.get()
        assert not result_path is None
        try:
            result_dict = json.load(open(result_path))
        except Exception as e:
            print("Experiment at {} failed".format(colored(result_path, 'red')))
            continue
        ## Only export keys we want to see in sheet to csv
        del_keys = []
        for key in result_dict:
            if not key in keys_to_display:
                del_keys.append(key)
        for key in del_keys:
            del result_dict[key]
        results.append( result_dict )
    results = sorted(results, key=lambda k: k['valid_f1'], reverse=True)

    dump_result_string = "Dumped grid search results to {}.\n".format(
        args.result_path, 
    )
    for k in keys_to_display:
        dump_result_string += "{} {}\n".format(k, results[0][k])

    with open(os.path.join(args.log_dir, args.result_path), 'w') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=keys_to_display )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(dump_result_string)
