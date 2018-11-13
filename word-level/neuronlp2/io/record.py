# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np

from efficiency.log import fwrite
import os
import pdb
import random
import json


class LossRecorder(object):
    def __init__(self, uid='temp'):
        self.__source_file = None
        self.uid = uid

    def start(self, file_path, mode='w', misc=''):
        self.__source_file = file_path
        self.__source_file2 = file_path + '.2'
        fwrite("# {}\n".format(misc), file_path, mode)
        fwrite("# {}\n".format(misc), self.__source_file2, mode)

    def write(self, epoch, loss1, loss2, lr, lr2, dev_f1, best_test_f1, test_f1):
        text = "{:03d} {:.4f} {:.2f} {:.2f} {:.2f}\n"\
            .format(epoch, loss1, dev_f1, best_test_f1, test_f1)
        fwrite(text, self.__source_file, mode='a')

        text = "ep:{:03d} loss1:{:.4f} loss2:{:.4f} lr:{:.2E}, lr2:{:.2E} dev_f1:{:.2f} b_test_f1:{:.2f} test_f1:{:.2f}\n"\
            .format(epoch, loss1, loss2, lr, lr2, dev_f1, best_test_f1, test_f1)
        fwrite(text, self.__source_file2, mode='a')

    def get_loss_list(self, file_path):
        list_loss = []
        list_dev = []
        list_test = []
        meta_line = 0
        with open(file_path) as f:
            for line_i, line in enumerate(f):
                if line.startswith("#"):
                    meta_line += 1
                    continue
                if not line.strip():
                    break
                line_i += 1
                epoch, loss, dev, test_best, *_ = [float(i) for i in line.split()]

                assert (line_i - meta_line) == epoch
                list_loss += [loss]
                list_dev += [dev]
                list_test += [test_best]
        return list_loss, list_dev, list_test


class LossPlot(object):
    def __init__(self, title='training', port=9999):
        self.viz = Visdom(port=port)
        self.viz.close()
        self.windows = {}
        self.title = title

    def plot_loss_trace(self, list_losses, loss_name):
        # self.viz.line(Y=np.random.rand(10), opts=dict(showlegend=True))

        max_len = max(len(loss) for loss in list_losses)
        Y = [np.zeros(max_len) for loss in list_losses]
        for i, loss in enumerate(list_losses):
            Y[i][:len(loss)] = np.array(loss)
        X = tuple(np.arange(len(i)) for i in Y)

        win = self.viz.line(
            Y=np.column_stack(Y),
            X=np.column_stack(X),
            opts=dict(title=self.title, markers=False, xlabel='epoch', ylabel=loss_name, showlegend=True)

        )
        self.windows[loss_name] = win
        import pdb;
        pdb.set_trace()


class TensorboardLossRecord(object):
    """docstring for TensorboardLossRecord"""

    def __init__(self, if_tensorboard, tensorboard_dir, uid='data'):
        super(TensorboardLossRecord, self).__init__()
        self.if_tensorboard = if_tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.uid = uid
        if if_tensorboard:
            from tensorboardX import SummaryWriter

            self.w = SummaryWriter(log_dir=tensorboard_dir)

    def plot_loss(self, epoch, tl, va):

        if self.if_tensorboard:
            self.w.add_scalars('{}/loss'.format(self.uid),
                               {'train': tl}, epoch)
            self.w.add_scalars('{}/accuracy'.format(self.uid),
                               {'valid': va}, epoch)

    def plot_text(self, epoch, text, title='Text'):
        if self.if_tensorboard:
            self.w.add_text(
                title, text, epoch)

    def plot_img(self, epoch, img, caption='Attention', image_name='', att_arr=None, xticks=[], yticks=[],
                 part=np.arange(10)):

        if self.if_tensorboard:
            if att_arr is not None:
                import torch
                if type(att_arr) is torch.Tensor:
                    att_arr = att_arr.detach().cpu().numpy()
                elif type(att_arr) is np.ndarray:
                    pass
                else:
                    assert False, "type of att_arr must be either numpy.ndarray or torch.Tensor"
                assert att_arr.ndim == 2
                # print("[Info]", json.dumps(xticks))

                att_arr = att_arr[part.reshape(-1, 1), part]
                xticks = [xticks[i] for i in part]
                yticks = [yticks[i] for i in part]
                img = heatmap2np(matrix=att_arr, xticks=xticks, yticks=yticks, xlabel='', ylabel='',
                                 title=caption[:40], image_name=image_name, decimals=8, # annt_clr='b',
                                 on_server=True)  # (480, 640, 4)

            try:
                self.w.add_image(
                    caption, img, epoch)
            except:
                img = np.transpose(img, (2, 0, 1))
                self.w.add_image(
                    caption, img, epoch)

    def close(self):
        if self.if_tensorboard:
            self.w.close()

    def visdo(self):
        from visdom import Visdom
        import tensorflow as tf
        from tensorflow.contrib.tensorboard.plugins import projector
        pass


def heatmap2np(matrix=None, xticks=[], yticks=[], xlabel='', ylabel='', title='', image_name="", decimals=1,
               annt_clr=None, on_server=True):
    import matplotlib
    if on_server:
        matplotlib.use('Agg')
    else:
        matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    # from pandas import DataFrame
    # import seaborn

    if (not (type(matrix) is np.ndarray)) and (not xticks) and (not yticks):
        yticks = ["cucumber", "tomato", "lettuce", "asparagus",
                  "potato", "wheat", "barley"]
        xticks = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
                  "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

        matrix = np.array([[0.823, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
                           [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
                           [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
                           [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
                           [0.7, 1.7, 0.624, 2.6, 2.2, 6.2, 0.0],
                           [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
                           [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])
        title = "Harvest of local xticks (in tons/year)"

    matrix = np.around(matrix, decimals=decimals)

    fig, ax = plt.subplots()

    im = ax.imshow(matrix, cmap='OrRd')

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(xticks)))
    ax.set_yticks(np.arange(len(yticks)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if annt_clr:
        for i in range(len(yticks)):
            for j in range(len(xticks)):
                text = ax.text(
                    j, i, matrix[i, j], ha="center", va="center", color=annt_clr)

    # data = DataFrame(data=matrix, columns=xticks, index=yticks)
    # data.columns.name = xlabel
    # data.index.name = ylabel
    # pdb.set_trace()

    # ax = seaborn.heatmap(data)

    fig.tight_layout()
    fig.canvas.draw()

    X = np.array(fig.canvas.renderer._renderer)
    # X = 0.2989 * X[:, :, 1] + 0.5870 * X[:, :, 2] + 0.1140 * X[:, :, 3] #
    # convert to black and white image

    if image_name:
        from PIL import Image
        X = X[:, :, :3] if image_name[-len(".jpg"):] == ".jpg" else X
        im = Image.fromarray(X)
        im.save(image_name)

    # plt.imshow(X, interpolation="none")
    # plt.show()
    plt.close('all')

    return X

    n = 3
    domain_size = 20

    x = np.random.randint(0, domain_size, (n, 2))

    fig, ax = plt.subplots()
    fig.set_size_inches((5, 5))
    ax.scatter(x[:, 0], x[:, 1], c="black", s=200, marker="*")
    ax.set_xlim(0, domain_size)
    ax.set_ylim(0, domain_size)
    fig.add_axes(ax)

    fig.canvas.draw()

    X = np.array(fig.canvas.renderer._renderer)
    X = 0.2989 * X[:, :, 1] + 0.5870 * X[:, :, 2] + 0.1140 * X[:, :, 3]
    show_var(["X", "x"])

    plt.imshow(X, interpolation="none", cmap="gray")
    plt.show()


if __name__ == "__main__":
    '''
    python -m visdom.server -p 9999
    '''
    from visdom import Visdom

    file_root = '/afs/csail.mit.edu/u/z/zhijing/proj/ie/data/run/'
    files = \
        ['aw/conl_loss',
         'aw/conl_loss_10250622r11',
         'aw/conl_loss_10250653r11',
         'az/03co_loss_10250741r10',  # /az just from /aw
         'az/03co_loss_10251206r10',  # /az just from /aw
         'az/03co_loss_10251505r11',  # /az 3-dim amendment
         'az/03co_loss_10251600r11',  # /az gcn god
         'az/03co_loss_10251642r11',  # /az gcn normal
         'az/03co_loss_11050002r10'  # /az std opt_schedule
         'az/03co_loss_11050257r10',  # /az std opt_schedule in batch
         'az/03co_loss_11060235r04',  # gcn tuning best
         'az/03co_loss_11062331r03'  # gcn with gold
         ]

    files = \
        [
            # 'az/03co_loss_10251505r11', # /az 3-dim amendment
            'az/03co_loss_11062331r03',  # gcn with gold
            'az/03co_loss_11070843r10',  # gcn with gold - debug
            'az/03co_loss_11071150r03',  # gcn with gold - debug
            'az/03co_loss_11060235r04'
            # ,  # gcn with repe best
            # 'az/03co_loss_temp'  # gcn with repe - debug
        ]

    n_files = len(files)
    list_loss, list_dev, list_test = list(range(n_files)), list(range(n_files)), list(range(n_files))

    for file_i, file_path in enumerate(files):
        file_path = file_root + file_path
        list_loss[file_i], list_dev[file_i], list_test[file_i] = LossRecorder().get_loss_list(file_path)

    import pdb;

    pdb.set_trace()
    plot = LossPlot("Model A")
    plot.plot_loss_trace(list_loss, "Loss")
    plot.plot_loss_trace(list_dev, "dev F1")
    plot.plot_loss_trace(list_test, "test F1")
