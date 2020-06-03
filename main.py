from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle
import random

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Doing this because eventually I will refactor everything to a v1, v2, v3 version of main and this will be a common function called in another 'header' file
def load_data():
	return tf.keras.datasets.fashion_mnist.load_data()

def batch(iterable, n=1):
	l = len(iterable)
	for ndx in range(0, l, n):
		yield iterable[ndx:min(ndx + n, l)]

def execute_agent(agent, cur_batch, data, labels, training=True):
	total_reward = 0
	for idx in cur_batch:
		agent.reset()
		guess = agent.act(data[idx])
		if guess == labels[idx]:
			total_reward += 1
	if training:
		agent.reward(total_reward)
	return total_reward

parser = ArgumentParser()
parser.add_argument('--version', type=int, default=1, help='Which version of the TPG you want to use')
args = parser.parse_args()

def main(args):
	(train_x, train_y), (test_x, test_y) = load_data()
	gens = 100
	rootTeamSize = 100
	batchSize = 1000
	version = 'v' + str(args.version)
	checkpoint_name = 'checkpoint_' + version + '.tpg'

	if version == 'v1':
		print('Using TPG Trainer V1 (no additional techniques)')
		from tpg_v1.trainer import Trainer
	elif version == 'v2':
		print('Using TPG Trainer V2 (shared registers)')
		from tpg_v2.trainer import Trainer
	elif version == 'v3':
		print('Using TPG Trainer V3 (shared registers with vector and matrix support)')
		from tpg_v3.trainer import Trainer
	elif version == 'v4':
		print('Using TPG Trainer V4 (shared registers with sub-observation indexing)')
		from tpg_v4.trainer import Trainer
	elif version == 'v5':
		print('Using TPG Trainer V5 (shared registers with multiple sub-observation indexing)')
		from tpg_v5.trainer import Trainer
	else:
		print('Please select a valid version')
		return 0

	if os.path.exists(checkpoint_name):
		print('Loading previous checkpoint')
		with open(checkpoint_name, 'rb') as f:
			temp = pickle.load(f)
			trainer = temp['trainer']
			gen = temp['gen']
			results = temp['results']
			del temp
	else:
		print('Making new checkpoint')
		trainer = Trainer(range(10), rootTeamSize, sourceRange=784)#, sourceDims=(28,28))
		gen = 1
		results = []

	#while gen < gens:
	while True:
		dataIdx = list(range(len(train_x)))
		random.shuffle(dataIdx)
		all_batches = [b for b in batch(dataIdx, n=batchSize)]
		for cur_batch in tqdm(all_batches, desc='Training batch', leave=False):
			agents = trainer.getAgents()
			for agent in agents:
				total_reward = 0
				for idx in cur_batch:
					agent.reset()
					guess = agent.act(train_x[idx])
					if guess == train_y[idx]:
						total_reward += 1
				agent.reward(total_reward)
			trainer.evolve()
		best_agent = None
		best_reward = 0
		agents = trainer.getAgents()
		for agent in tqdm(agents, desc='Testing generation: {}'.format(gen), leave=False):
			agent_reward = 0
			for idx in range(len(test_x)):
				agent.reset()
				guess = agent.act(test_x[idx])
				if guess == test_y[idx]:
					agent_reward += 1
			if best_agent is None or agent_reward > best_reward:
				best_agent = agent.agentNum
				best_reward = agent_reward
		print('Gen {}, Agent #{}, Reward: {}/{}'.format(gen, best_agent, best_reward, len(test_x)))
		results.append([gen, best_reward])
		gen += 1
		with open(checkpoint_name, 'wb') as f:
			pickle.dump({'trainer': trainer, 'gen': gen, 'results': results}, f)
		


if __name__ == '__main__':
	main(args)