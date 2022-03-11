import pickle

import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import experiments

def plot_CartPole_Q_function(Q_network, figure_left, figure_right, figure_diff, shared_title=''):
    x_coord = np.linspace(-2, 2, 5)
    v_coord = np.linspace(-2, 2, 5)
    theta_coord = np.linspace(-0.2, 0.2, 101)
    omega_coord = np.linspace(-1, 1, 101)

    x, v, theta, omega = np.meshgrid(x_coord, v_coord, theta_coord, omega_coord)

    obs_vector = np.stack([x, theta, v, omega], axis=-1).reshape(-1, 4)

    Q_space_shape = x.shape
    Q_values = np.array(Q_network.keras_network(obs_vector)).reshape(Q_space_shape+(2,))

    min_Q = np.min(Q_values)
    max_Q = np.max(Q_values)

    max_Q_diff = np.max(np.abs(Q_values[...,1] - Q_values[...,0]))

    for fig in [figure_left, figure_right, figure_diff]:
        plt.figure(fig)

        for row in range(5):
            for col in range(5):
                plt.subplot(5, 5, 1+row*5+col)

                if row == 4 and col == 2:
                    plt.xlabel(f'$\\theta$\n\n{round(x_coord[col])}\n---\n$x$')
                elif row == 4:
                    plt.xlabel(f'$~$\n\n{round(x_coord[col])}\n---')
                    for i in plt.gca().get_xticklabels():
                        i.set_color('w')
                else:
                    plt.gca().axes.xaxis.set_visible(False)

                if col == 0 and row == 2:
                    plt.ylabel(f'$v$\n---\n{round(v_coord[4-row])}\n\n$\\omega$')
                elif col == 0:
                    plt.ylabel(f'---\n{round(v_coord[4-row])}\n\n$~$')
                    for i in plt.gca().get_yticklabels():
                        i.set_color('w')
                else:
                    plt.gca().axes.yaxis.set_visible(False)

                if fig is figure_left:
                    array = Q_values[col, 4-row][:,:,0]
                    cmap = 'turbo'
                    vmin = min_Q
                    vmax = max_Q
                elif fig is figure_right:
                    array = Q_values[col, 4-row][:,:,1]
                    cmap = 'turbo'
                    vmin = min_Q
                    vmax = max_Q
                else:
                    array = Q_values[col, 4-row][:,:,1] - Q_values[col, 4-row][:,:,0]
                    cmap = 'bwr'
                    vmin = -max_Q_diff
                    vmax = max_Q_diff

                im = plt.imshow(array, aspect='auto', origin='lower', extent=(-0.2, 0.2, -1, 1), vmin=vmin, vmax=vmax, cmap=cmap)
                plt.ylim(-1, 1)
                plt.xlim(-0.2, 0.2)

                if fig is figure_diff:
                    plt.contour(array, origin='lower', extent=(-0.2, 0.2, -1, 1), levels=[0], colors=['black'], linestyles=['--'], linewidths=[0.5])

        if fig is figure_left:
            plt.suptitle(shared_title+'Quality of left action')
        elif fig is figure_right:
            plt.suptitle(shared_title+'Quality of right action')
        else:
            plt.suptitle(shared_title+'Difference of action qualities (right - left)')

        fig.subplots_adjust(left=0.18, bottom=0.22, right=0.8, hspace=0.05, wspace=0.05)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

def plot_CNN_weights(best_weights, title=None):
    dynamics_filters = best_weights[0]
    emergency_filters = best_weights[2]

    passthru = (best_weights[4] @ best_weights[6] @ best_weights[8])
    dynamics_passthru = passthru[:-12].reshape(10,10,12,2)
    emergency_passthru = passthru[-12:]

    location_sensitivity = np.std(dynamics_passthru[:,:,:,1] - dynamics_passthru[:,:,:,0], axis=(0,1))
    emergency_sensitivity = emergency_passthru[:,1] - emergency_passthru[:,0]

    loc_up = np.argmin(dynamics_passthru[:,:,:,1] - dynamics_passthru[:,:,:,0])
    location_up_feature = loc_up % 12
    loc_up_x = loc_up // 12 % 10
    loc_up_y = loc_up // 12 // 10

    loc_down = np.argmax(dynamics_passthru[:,:,:,1] - dynamics_passthru[:,:,:,0])
    location_down_feature = loc_down % 12
    loc_down_x = loc_down // 12 % 10
    loc_down_y = loc_down // 12 // 10

    strongest_down_feature = np.argmax(emergency_sensitivity)
    strongest_up_feature = np.argmin(emergency_sensitivity)

    strongest_location_features = np.argsort(-location_sensitivity)[:2]

    source_names = ['BALL(t)', 'PLR(t)', 'OPP(t)', 'B(t-3)', 'P(t-3)', 'O(t-3)']

    filter_names = ['EM\nDOWN', 'EM\nUP', 'LOC\n(\\#1)', 'LOC\n(\\#2)', f'DYN UP\n(@ {loc_up_x}, {loc_up_y})', f'DYN DOWN\n(@ {loc_down_x}, {loc_down_y})']
    filter_ids = [strongest_up_feature, strongest_down_feature, strongest_location_features[0], strongest_location_features[1], location_up_feature, location_down_feature]
    filter_types = [emergency_filters, emergency_filters, dynamics_filters, dynamics_filters, dynamics_filters, dynamics_filters]
    n_cols = len(filter_names)

    vmax_d = np.max(np.abs(dynamics_filters))
    vmax_e = np.max(np.abs(emergency_filters))
    vmax = max(vmax_d, vmax_e)

    plt.figure()
    for row in range(6):
        for col, filter_name, filter_id, filter_type in zip(range(n_cols), filter_names, filter_ids, filter_types):
            plt.subplot(6, n_cols, 1+row*n_cols+col)

            if col == 0:
                plt.ylabel(source_names[row])
            else:
                plt.gca().axes.yaxis.set_visible(False)
            if row == 5:
                plt.xlabel(f'{filter_name}: {filter_id}')
            else:
                plt.gca().axes.xaxis.set_visible(False)

            plt.gca().axes.xaxis.set_ticks([])
            plt.gca().axes.yaxis.set_ticks([])

            plt.imshow(filter_type[:,:,row,filter_id], vmin=-vmax, vmax=vmax, cmap='bwr')

    if title is None:
        title = 'CNN'

    plt.suptitle(f'Most relevant filters of {title} agent')

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

def timesteps_to_episode(timesteps, episodes):
    timesteps = np.array(timesteps)
    episode_indices = np.array([0]+[ts for ts, _ in episodes])

    return np.digitize(timesteps, episode_indices, right=True)

def plot_training_curves(records, suptitle='Training progress', out_fn=None):
    for record in records:
        print(len(record['episode_rewards']))
    episode_lengths = [[t2 - t1 for (t2, r2), (t1, r1) in zip(record['episode_rewards'][1:], record['episode_rewards'][:-1])] for record in records]
    episode_returns = [[r for t, r in record['episode_rewards']] for record in records]
    behavior_entropys = [[e for t, e in record['episode_behavior_entropy']] for record in records]
    loss_sampless = [[l for t, l in record['loss_samples']] for record in records]
    loss_timesteps = [[t for t, l in record['loss_samples']] for record in records]

    plt.figure()
    for i, title, runs in zip([1,2,3,4], ['100-episode return', 'Episode length', 'Behavior entropy', 'Sample loss (RMSE)'], [episode_returns, episode_lengths, behavior_entropys, loss_sampless]):
        smoothed_runs = [last_n_average(run, 100) for run in runs]
        plt.subplot(2,2,i)
        plt.title(title+' (running mean)')

        plot_losses = False

        if runs is loss_sampless:
            plot_losses = True
            loss_i = 0
            episode_indices = [timesteps_to_episode(loss_timestep, record['episode_rewards']) for loss_timestep, record in zip(loss_timesteps, records)]
            plt.gca().set_yscale('log')
        elif len(runs) > 1:
            if runs is episode_lengths:
                expected_length = 2499
            else:
                expected_length = 2500
            runs_to_include = [i for i in smoothed_runs if i.size == expected_length]
            mean_run = np.mean(runs_to_include, axis=0)

            run_l, run_h = st.t.interval(0.95, len(runs_to_include)-1, loc=mean_run, scale=st.sem(runs_to_include, axis=0))

            plt.fill_between(np.arange(run_l.size), run_l, run_h, color='black', alpha=0.25)
            plt.plot(run_l, color='black', lw=0.5, ls='--')
            plt.plot(run_h, color='black', lw=0.5, ls='--', label='95\\% c.i.')

        label = (i == 2)
        for run in smoothed_runs:
            if plot_losses:
                plt.plot(episode_indices[loss_i], run, alpha=0.75)
                loss_i += 1
            elif len(runs) == 1:
                plt.plot(run, label='indiv. run')
            elif label:
                plt.plot(run, alpha=0.25, label='indiv. run')
            else:
                plt.plot(run, alpha=0.25)
            label = False

        if not plot_losses and len(runs) > 1:
            plt.plot(mean_run, color='red', label='mean run')

        if i == 2:
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

        if i > 2:
            plt.xlabel('Episode')

    plt.suptitle(suptitle)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.78, hspace=0.3, wspace=0.25)

    if not out_fn is None:
        plt.savefig(out_fn)
        plt.close()

def report_performance(agent_fn, task_config, seeds, *args, **kwargs):
    mean_episode_returns = []
    for seed in seeds:
        sim = agent_fn(seed, task_config, *args, replay=True, episodes=10, **kwargs)
        mean_episode_returns.append(np.mean([ret for ts, ret in sim.episode_rewards]))
    return np.mean(mean_episode_returns), np.std(mean_episode_returns), mean_episode_returns

def report_cartpole_performance(seeds_MC, seeds_FQI, seeds_DQN):
    MC_cartpole_perfs = report_performance(experiments.test_MC_Agent, experiments.cart_pole_config, seeds_MC)[2]
    FQI_cartpole_perfs = report_performance(experiments.test_FQI_Agent, experiments.cart_pole_config, seeds_FQI)[2]
    DQN_cartpole_perfs = report_performance(experiments.test_DQN_Agent, experiments.cart_pole_config, seeds_DQN)[2]

    return MC_cartpole_perfs, FQI_cartpole_perfs, DQN_cartpole_perfs

def plot_cartpole_performance(perfs):
    MC_cartpole_perfs, FQI_cartpole_perfs, DQN_cartpole_perfs = perfs

    plt.figure()
    plt.title('Per-run best performance on cart-pole task')
    plt.violinplot(perfs, showmeans=True)
    plt.gca().xaxis.set_ticks([1,2,3],['MC','FQI','DQN'])
    plt.ylabel('10-episode mean return')

    plt.savefig('plots/cartpole-performance.pdf')
    plt.close()

def plot_cartpole_Q_function(a, seed, title=None, out_prefix=None):
    task_name, _, _, Q_network, _, t = experiments.cart_pole_config(np.random.default_rng(seed+234579672983459873))
    record = pickle.load(open(f'out/{a}-{task_name}-{seed}.pickle', 'rb'))
    Q_network.keras_network.set_weights(record['best_weights'])

    if title is None:
        title = f'{a}-{seed}'

    figure_left = plt.figure()
    figure_right = plt.figure()
    figure_diff = plt.figure()
    plot_CartPole_Q_function(Q_network, figure_left, figure_right, figure_diff, title+'\n')

    if out_prefix:
        figure_left.savefig(out_prefix+'.left.pdf')
        figure_right.savefig(out_prefix+'.right.pdf')
        figure_diff.savefig(out_prefix+'.diff.pdf')
        plt.close(figure_left)
        plt.close(figure_right)
        plt.close(figure_diff)

def plot_best_cartpole_Q_functions(seeds, perfs):
    for seed_set, perf_set, label in zip(seeds, perfs, ["MC", "FQI", "DQN"]):
        seed = seed_set[np.argmax(perf_set)]

        plot_cartpole_Q_function(label, seed, title=f'Best {label} run ({seed})', out_prefix=f'plots/Q_net-cartpole-{label}-{seed}')

def plot_cartpole_training_curves(seeds, label):
    records = load_records(f'{label}-CartPole', seeds)

    plot_training_curves(records, suptitle=f'{label} training progress on cart-pole task', out_fn=f'plots/training-cartpole-{label}.pdf')

def plot_FQI_random_actions():
    plot_cartpole_Q_function('FQI-RandomActions', 1, 'FQI with random actions', out_prefix=f'plots/Q_net-cartpole-FQI-RandomActions')
    plot_cartpole_training_curves([1], 'FQI-RandomActions')

def plot_pong_training_curves(seeds, label):
    records = load_records(f'{label}-Pong', seeds)

    plot_training_curves(records, suptitle=f'{label} training progress on pong task', out_fn=f'plots/training-pong-{label}.pdf')

def plot_pong_weights(a, seed, title=None, out_fn=None):
    if title is None:
        title = f'{a}-{seed}'

    record = pickle.load(open(f'out/{a}-Pong-{seed}.pickle', 'rb'))
    plot_CNN_weights(record['best_weights'], title)

    if out_fn:
        plt.savefig(out_fn)
        plt.close()

def plot_all_pong_weights():
    for a in ['MC', 'FQI', 'DQN']:
        plot_pong_weights(a, 1, a, f'plots/features-pong-{a}.pdf')

if __name__ == '__main__':
    import sys
    a = sys.argv[1]
    t = sys.argv[2]
    seed = int(sys.argv[3])

    if t == 'pong':
        tas = experiments.pong_config
    elif t == 'cartpole':
        tas = experiments.cart_pole_config
    else:
        print('invalid task')
        sys.exit()

    task_name, _, _, Q_network, _, t = tas(np.random.default_rng(seed+234579672983459873))

    record = pickle.load(open(f'out/{a}-{task_name}-{seed}.pickle', 'rb'))
    Q_network.keras_network.set_weights(record['best_weights'])

    plot_CartPole_Q_function(Q_network, plt.figure(), plt.figure(), plt.figure())
    plt.show()
