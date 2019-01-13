import numpy as np
import matplotlib.pyplot as plt

from bandits.core.contextual_bandit import run_contextual_bandit

import pickle, time, sys, os

class Benchmarker(object):
    """
    Takes functions that create algos and dataset so as to rerun experiments several times and plot results.
    """

    def __init__(self, algo_protos, dataset_proto, num_actions, context_dim, nb_contexts, test_name):
        self.algo_protos = algo_protos
        self.dataset_proto = dataset_proto
        self.test_name = test_name
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.nb_contexts = nb_contexts

    def run_experiments(self, iterations = 10):
        cum_rew = np.zeros((self.nb_contexts, len(self.algo_protos), iterations))
        cum_reg = np.zeros(cum_rew.shape)

        for iter in range(iterations):
            print(str(iter+1),'/',str(iterations))
            t_init = time.time()

            dataset, opt_linear = self.dataset_proto()
            print('dataset created')
            opt_rewards, opt_actions = opt_linear

            algos = [algo_proto() for algo_proto in self.algo_protos]
            print('algo ready')

            outcome = run_contextual_bandit(self.context_dim, self.num_actions, dataset, algos)
            h_actions, h_rewards = outcome

            cum_rew[:,:,iter] = np.cumsum(h_rewards, axis=0)
            cum_reg[:,:,iter] = np.cumsum(opt_rewards)[:,np.newaxis] - cum_rew[:,:,iter]

            # print('Iter {} took {} ms'%(iter, time.time()-t_init))

        # if other_results is not None:
        #     self.results = np.concatenate((other_results, results), axis=2)
        # else:
        #     self.results = results
        self.cum_rew = cum_rew
        self.cum_reg = cum_reg

    def save_results(self, path, prefix = ''):
        algos = [algo_proto() for algo_proto in self.algo_protos]
        dic = {
            'test_name': self.test_name,
            'num_actions': self.num_actions,
            'context_dim': self.context_dim,
            'nb_contexts': self.nb_contexts,
            'cum_rew': self.cum_rew,
            'cum_reg': self.cum_reg,
            'algo_details': str([(algo.name, algo.hparams.to_json) for algo in algos])
        }
        #, 'dataset_proto': self.dataset_proto, 'algo_protos': self.algo_protos

        with open(path + prefix + '_' + self.test_name + '.pickle', 'wb') as handle:
            pickle.dump(dic, handle)

    # def load(path):
    #     with open(path, 'rb') as handle:
    #         dic = pickle.read(dic, handle)
    #     benchmarker = Benchmarker(dic['algo_protos'], dic['dataset_proto'], dic['num_actions'], dic['context_dim'], dic['nb_contexts'], dic['test_name'])
    #     benchmarker.results = dic['results']
    #     return benchmarker

    def display_results(self, save_path=None):
        plt.figure()
        algos = [algo_proto() for algo_proto in self.algo_protos]
        algo_names = [algo.name for algo in algos]


        t = np.arange(self.cum_reg.shape[0])
        res = self.cum_reg
        means, stds =  np.mean(res, axis=2), np.std(res, axis=2)
        for i, algo_name in enumerate(algo_names):
            mean, std = means[:,i], stds[:,i]
            plt.plot(t, mean, label=algo_name)
            plt.fill_between(t, mean-std, mean+std, alpha=0.3)
        plt.xlabel('Step')
        plt.ylabel('Cumulative regret')
        plt.legend()
        plt.title(self.test_name+' nb Runs : %i, n_d=%i, n_a=%i, t=%i'%(res.shape[2], self.context_dim, self.num_actions, self.nb_contexts))
        if save_path is not None:
            plt.savefig(save_path+self.test_name+'.png')
        plt.show()
