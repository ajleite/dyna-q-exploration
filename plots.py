import pickle

import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt

def last_n_average(values, n):
    values = np.array(values)
    n = min(n, values.size)
    num = np.cumsum(values)
    den = np.cumsum(np.ones_like(values))
    prev_num = np.concatenate([np.zeros(n), num[:-n]], axis=0)
    prev_den = np.concatenate([np.zeros(n), den[:-n]], axis=0)
    return (num-prev_num)/(den-prev_den)

def load_records(condition, seeds):
    records = []
    for s in seeds:
        records.append(pickle.load(open(f'out/{condition}-{s}.pickle', 'rb')))
    return records

def plot_training_curves(records, suptitle='Training progress', out_fn=None):
    training_returns = [record['training_episode_rewards'] for record in records]
    evaluation_returns = [[np.mean(r) for r in record['eval_episode_rewards']] for record in records]
    Q_loss = [record['Q_rmse'] for record in records]
    dyn_loss = [record['dyn_rmse'] for record in records]
    training_cycle_length = 50
    training_cycle_count = len(records[0]['training_episode_rewards'])/50
    training_episode_count = len(records[0]['training_episode_rewards'])

    training_episode_numbers = np.arange(0, training_episode_count)

    eval_cycle_length = 50

    eval_episode_numbers = records[0]['eval_indices']

    plt.figure()
    for i, title, x_axis, runs in zip([1,2,3,4], ['Training return (running mean)', 'Evaluation return', 'Q loss (RMSE)', 'Dynamics loss'], [training_episode_numbers, eval_episode_numbers, training_episode_numbers, training_episode_numbers], [training_returns, evaluation_returns, Q_loss, dyn_loss]):
        if not runs[0]:
            continue

        if runs is training_returns:
            runs = [last_n_average(run, 100) for run in runs]
            # title = title+' (running mean)'

        run_length = np.min([len(run) for run in runs])
        runs = [run[:run_length] for run in runs]
        x_axis = x_axis[:run_length]

        plt.subplot(2,2,i)
        plt.title(title)

        if len(runs) > 1:
            mean_run = np.mean(runs, axis=0)

            run_l, run_h = st.t.interval(0.95, len(runs)-1, loc=mean_run, scale=st.sem(runs, axis=0))

            plt.fill_between(x_axis, run_l, run_h, color='black', alpha=0.25)
            plt.plot(x_axis, run_l, color='black', lw=0.5, ls='--')
            plt.plot(x_axis, run_h, color='black', lw=0.5, ls='--', label='95% c.i.')

        label = (i == 2)
        for run in runs:
            if len(runs) == 1:
                plt.plot(x_axis, run, label='indiv. run')
            elif label:
                plt.plot(x_axis, run, alpha=0.25, label='indiv. run')
            else:
                plt.plot(x_axis, run, alpha=0.25)
            label = False

        if len(runs) > 1:
            plt.plot(x_axis, mean_run, color='red', label='mean run')

        if i == 2:
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

        if i > 2:
            plt.xlabel('Episode')

    plt.suptitle(suptitle)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.78, hspace=0.3, wspace=0.25)

    if not out_fn is None:
        plt.savefig(out_fn)
        plt.close()

def plot_performance(DQN_records, DynaQ_records, task_name, out_fn=None):
    DQN_perfs = [record['best_100_episode_return'] for record in DQN_records]
    DynaQ_perfs = [record['best_100_episode_return'] for record in DynaQ_records]

    print(f"DQN on {task_name}: {np.mean(DQN_perfs)} +/- {np.std(DQN_perfs)}")
    print(f"DynaQ on {task_name}: {np.mean(DynaQ_perfs)} +/- {np.std(DynaQ_perfs)}")

    plt.figure()
    plt.title(f'Per-run best performance on {task_name} task')
    plt.violinplot([DQN_perfs, DynaQ_perfs], showmeans=True)
    plt.gca().xaxis.set_ticks([1,2],['DQN','DynaQ'])
    plt.ylabel('100-episode mean return')

    if not out_fn is None:
        plt.savefig(out_fn)
        plt.close()

if __name__ == '__main__':
    plot_training_curves(load_records("DQN-Pong-PostCNN", range(1,2)))
    plt.show()
