from __future__ import division, print_function

from torch.optim import Adam, SGD

class Optimizer(object):

    def __init__(self, optimizer_type_lstm, optimizer_type_gcn, network, dropout, lr=0.01, lr_gcn=0.01,
                 wd=0., wd_gcn=0., momentum=0.9, lr_decay=0.05, schedule=1, gcn_warmup=2000, pretrain_lstm=0):
        optimizer_type = {'sgd': SGD, 'adam': Adam}
        self.optimizer_type_lstm = optimizer_type[optimizer_type_lstm]
        self.optimizer_type_gcn = optimizer_type[optimizer_type_gcn]
        self.dropout = dropout
        self.lr = lr
        self.lr_gcn = lr_gcn
        self.wd = wd
        self.wd_gcn = wd_gcn
        self.schedule = schedule
        self.momentum = momentum
        self.lr_decay = lr_decay

        self.gcn_warmup = gcn_warmup
        self.pretrain_lstm = pretrain_lstm

        lstm_params = [{'params': [param]} for name, param in network.named_parameters() if 'gcn' not in name]
        lstm_names = [name for name, param in network.named_parameters() if 'gcn.gcn_layers' not in name]
        gcn_names = [name for name, param in network.named_parameters() if 'gcn.gcn_layers' in name]
        gcn_params = [{'params': [param]} for name, param in network.named_parameters() if 'gcn' in name]
        self.opt_lstm = self.optimizer_type_lstm(lstm_params,
                                                 lr=lr, weight_decay=wd, momentum=momentum, nesterov=True)
        if dropout == 'gcn':
            self.opt_gcn = self.optimizer_type_gcn(gcn_params,
                                               lr=0., weight_decay=wd_gcn)
        import pdb; pdb.set_trace()
        self.curr_lr = lr
        self.warmth = 0.
        self.curr_lr_gcn = 0.

    def update(self, epoch, batch, num_batches, network, debug=False):
        pretrain_lstm = self.pretrain_lstm
        lr = self.lr
        lr_gcn = self.lr_gcn if epoch > pretrain_lstm else 0
        wd = self.wd
        wd_gcn = self.wd_gcn
        momentum = self.momentum
        lr_decay = self.lr_decay
        schedule = self.schedule
        dropout = self.dropout

        if epoch % schedule == 0:
            lr *= 1 / (1.0 + epoch * lr_decay)

        if debug:
            import pdb; pdb.set_trace()
        # get warmth
        gcn_warmup = self.gcn_warmup
        step = max(0, (epoch - 1 - pretrain_lstm) * num_batches + batch)
        if step <= gcn_warmup:
            warmth = step / gcn_warmup
        else:
            warmth = 1 / (1.0 + epoch * lr_decay)

        self.adjust_learning_rate(self.opt_lstm, lr)

        if dropout == 'gcn':
            lr_gcn *= warmth
            self.adjust_learning_rate(self.opt_gcn, lr_gcn)

            # self.opt_gcn = self.optimizer_type_gcn([
            #     {'params': gcn_params, 'weight_decay': wd_gcn}],
            #     lr=lr_gcn, weight_decay=wd_gcn)

            # self.opt_lstm.add_param_group(
            #     {"params": network.rnn2.parameters(), 'lr': lr_gcn})

        self.curr_lr = lr
        self.warmth = warmth
        self.curr_lr_gcn = lr_gcn

    def adjust_learning_rate(self, optimizer, lr):
        """Sets the learning rate"""
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def step(self):
        dropout = self.dropout
        self.opt_lstm.step()

        if dropout == 'gcn':
            self.opt_gcn.step()

    def zero_grad(self):
        dropout = self.dropout
        self.opt_lstm.zero_grad()

        if dropout == 'gcn':
            self.opt_gcn.zero_grad()

    def state_dict(self):
        dropout = self.dropout

        dic = self.opt_lstm.state_dict()
        if dropout == 'gcn':
            dic.update(self.opt_gcn.state_dict())
            return dic