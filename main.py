import experiments
import plots

def generate_agents():
	for seed in range(1, 11):
		experiments.test_MC_Agent(seed, experiments.cart_pole_config)
		experiments.test_FQI_Agent(seed, experiments.cart_pole_config)
		experiments.test_DQN_Agent(seed, experiments.cart_pole_config)

	experiments.test_FQI_Agent(1, experiments.cart_pole_config, random_actions=True)

	experiments.test_MC_Agent(1, experiments.pong_config)
	experiments.test_FQI_Agent(1, experiments.pong_config)
	experiments.test_DQN_Agent(1, experiments.pong_config)

def evaluate_agents():
	MC_cartpole_performance = plots.report_performance(experiments.test_MC_Agent, experiments.cart_pole_config, range(1, 11))
	FQI_cartpole_performance = plots.report_performance(experiments.test_FQI_Agent, experiments.cart_pole_config, range(1, 11))
	DQN_cartpole_performance = plots.report_performance(experiments.test_DQN_Agent, experiments.cart_pole_config, range(1, 11))

	FQI_random_actions_performance = plot.report_performance(experiments.test_FQI_Agent, experiments.cart_pole_config, [1], random_actions=True)

	MC_pong_performance = plots.report_performance(experiments.test_MC_Agent, experiments.pong_config, [1])
	FQI_pong_performance = plots.report_performance(experiments.test_FQI_Agent, experiments.pong_config, [1])
	DQN_pong_performance = plots.report_performance(experiments.test_DQN_Agent, experiments.pong_config, [1])

	print('Cart-pole performance:')
	print(f'MC: {MC_cartpole_performance[0]:.2f} +/- {MC_cartpole_performance[1]:.2f}')
	print(f'FQI: {FQI_cartpole_performance[0]:.2f} +/- {FQI_cartpole_performance[1]:.2f}')
	print(f'DQN: {DQN_cartpole_performance[0]:.2f} +/- {DQN_cartpole_performance[1]:.2f}')

	print(f'FQI with random actions: {FQI_random_actions_performance[0]:.2f}')

	print('Pong performance:')
	print(f'MC: {MC_pong_performance[0]:.2f}')
	print(f'FQI: {FQI_pong_performance[0]:.2f}')
	print(f'DQN: {DQN_pong_performance[0]:.2f}')

def generate_plots():
	seeds = [range(1, 11), range(1, 11), range(1, 11)]
	perfs = plots.report_cartpole_performance(*seeds)
	plots.plot_cartpole_performance(perfs)
	plots.plot_best_cartpole_Q_functions(seeds, perfs)

	for seed_set, label in zip(seeds, ['MC', 'FQI', 'DQN']):
		plots.plot_cartpole_training_curves(seed_set, label)

	for label in ['MC', 'FQI', 'DQN']:
		plots.plot_pong_training_curves([1], label)

	plots.plot_all_pong_weights()

if __name__ == '__main__':
	generate_agents()
	evaluate_agents()
	generate_plots()