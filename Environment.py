import tensorflow
import numpy
import pandas
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
from math import ceil
from Models.TransE import TransE
from Models.TransH import TransH
from Models.TransR import TransR
from Models.TransD import TransD
from Datasets.Countries import Countries
from Datasets.Kinship import Kinship
from Datasets.UMLS import UMLS
from Datasets.FB15K import FB15K
from Datasets.NELL995 import NELL995

class Environment():

	def __init__(self, dataset = None, load_environment = None, train_environment = None, save_environment = None, number_of_steps = None, learning_rate = None, plot_loss_curves = None, save_figure = None, evaluate_environment = None, number_of_walks = None, walk_lengths = None, summarize_results = None):

		self.dataset = dataset
		self._load_environment = load_environment
		self._save_environment = save_environment
		self.number_of_walks = number_of_walks
		self.walk_lengths = walk_lengths
		self.learning_rate = learning_rate
		self.number_of_steps = number_of_steps
		self.save_figure = save_figure

		embedding_dim = self.dataset.embedding_size

		self.head_entity_embedding_vector = tensorflow.placeholder(tensorflow.float32)
		self.relation_embedding_vector = tensorflow.placeholder(tensorflow.float32)
		self.target_entity_embedding_vector = tensorflow.placeholder(tensorflow.float32)
		self.candidate_entity_embedding_vector = tensorflow.placeholder(tensorflow.float32)
		
		relation_embedding_vector = tensorflow.reshape(self.relation_embedding_vector, (-1, embedding_dim))

		in0 = tensorflow.keras.Input(tensor = relation_embedding_vector)
		self.dense1 = tensorflow.keras.layers.Dense(units = embedding_dim * embedding_dim, activation = 'linear') 
		out0 = self.dense1(in0)

		relation_transform = tensorflow.reshape(out0, (-1, embedding_dim, embedding_dim))

		head_entity_embedding_vector = tensorflow.reshape(self.head_entity_embedding_vector, (-1, embedding_dim, 1))
		intermediate_entity = tensorflow.matmul(relation_transform, head_entity_embedding_vector)

		intermediate_entity = tensorflow.reshape(intermediate_entity, (-1, embedding_dim))
		target_entity_embedding_vector = tensorflow.reshape(self.target_entity_embedding_vector, (-1, embedding_dim))

		in1 = tensorflow.keras.Input(tensor = intermediate_entity)
		in2 = tensorflow.keras.Input(tensor = target_entity_embedding_vector)
		net = tensorflow.keras.layers.Concatenate()([in0, in1, in2])
		self.dense2 = tensorflow.keras.layers.Dense(units = embedding_dim, activation = 'linear') 
		out1 = self.dense2(net)

		towards_target_transform = tensorflow.reshape(out1, (-1, embedding_dim, 1))
		intermediate_entity = tensorflow.reshape(intermediate_entity, (-1, embedding_dim, 1))
		
		predict_tail_entity_embedding_vector = tensorflow.add(intermediate_entity, towards_target_transform)

		head_entity_embedding_vector = tensorflow.reshape(head_entity_embedding_vector, (-1, embedding_dim))
		predict_tail_entity_embedding_vector = tensorflow.reshape(predict_tail_entity_embedding_vector, (-1, embedding_dim))
		target_entity_embedding_vector = tensorflow.reshape(target_entity_embedding_vector, (-1, embedding_dim))

		self.predict_tail_entity_embedding_vector = predict_tail_entity_embedding_vector
		
		loss_jump = tensorflow.reduce_sum(abs(tensorflow.linalg.norm((target_entity_embedding_vector - predict_tail_entity_embedding_vector), 1, keep_dims = True) - tensorflow.linalg.norm((target_entity_embedding_vector - head_entity_embedding_vector), 1, keep_dims = True)), 0, keep_dims = True)

		self.loss_jump = loss_jump

		loss_valid = tensorflow.reduce_sum(tensorflow.reduce_sum((self.candidate_entity_embedding_vector - predict_tail_entity_embedding_vector)**2, 1, keep_dims = True), 0, keep_dims = True)

		self.loss_valid = loss_valid

		self.loss = self.loss_jump + self.loss_valid
		
		self.global_step = tensorflow.Variable(0, name = "global_step", trainable = False)

		self.optimizer = tensorflow.train.AdamOptimizer(learning_rate = self.learning_rate)
		self.gradients = self.optimizer.compute_gradients(self.loss)
		self.learn = self.optimizer.apply_gradients(self.gradients, global_step = self.global_step)

		self.history_loss_jump = numpy.array([])
		self.history_loss_valid = numpy.array([])
		self.history_loss = numpy.array([])
		self.test_data = []

		if load_environment:
			self.load_environment()

		if train_environment:
			self.train_environment()
		
		if plot_loss_curves:
			self.plot_loss_curves()

		if evaluate_environment:
			self.evaluate_environment()

		if save_environment:
			self.save_environment()

		if summarize_results:
			self.summarize_results()

	def train_environment(self):

		session = tensorflow.get_default_session()
		
		_entity_ids = numpy.arange(self.dataset.entity_count)
		_relation_ids = numpy.arange(self.dataset.relation_count)

		session.run(tensorflow.global_variables_initializer())
		self.trainable_variables = numpy.sum([numpy.prod(v.get_shape().as_list()) for v in tensorflow.trainable_variables()])

		print("\nTotal trainable variables =", self.trainable_variables)
		print()

		history_loss_jump = []
		history_loss_valid = []
		history_loss = []

		number_of_steps = self.number_of_steps
		batch_size = self.dataset.batch_size

		for i in tqdm(range(ceil(number_of_steps//batch_size))):
	
			head_ids = []
			relation_ids = []
			target_ids = []
			candidate_ids = []

			batch_count = 0
			
			while (batch_count < batch_size) and ((i*batch_size + batch_count) < number_of_steps): 

				head_id = numpy.random.choice(_entity_ids)
				relation_id = numpy.random.choice(_relation_ids)
				target_id = numpy.random.choice(_entity_ids)
			
				candidate_triplets = self.dataset.triplets[self.dataset.triplets[:, 0] == head_id]
				
				if not len(candidate_triplets):
					candidate_id = target_id
	
				else:		
					_candidate_triplets = candidate_triplets[candidate_triplets[:, 1] == relation_id]

					if len(_candidate_triplets):
						candidate_triplets = _candidate_triplets

					_candidate_ids = candidate_triplets[:, 2]
					candidate_id = min(_candidate_ids, key = lambda x: numpy.linalg.norm(self.dataset.entityid2vec[target_id] - self.dataset.entityid2vec[x]))
					
				head_ids.append(head_id)
				relation_ids.append(relation_id)
				target_ids.append(target_id)
				candidate_ids.append(candidate_id)

				batch_count += 1

			head_entity_embedding_vectors = self.dataset.entityid2vec[head_ids]
			relation_embedding_vectors = self.dataset.relationid2vec[relation_ids]
			target_entity_embedding_vectors = self.dataset.entityid2vec[target_ids]
			candidate_entities_embedding_vectors = self.dataset.entityid2vec[candidate_ids]

			feed_dict = {
							self.head_entity_embedding_vector: head_entity_embedding_vectors,
							self.relation_embedding_vector: relation_embedding_vectors,
							self.target_entity_embedding_vector: target_entity_embedding_vectors,
							self.candidate_entity_embedding_vector: candidate_entities_embedding_vectors
						}

			_, step, loss_jump, loss_valid, loss = session.run([self.learn, self.global_step, self.loss_jump, self.loss_valid, self.loss], feed_dict = feed_dict)

			history_loss_jump.append(loss_jump)
			history_loss_valid.append(loss_valid)
			history_loss.append(loss)

		history_loss_jump = numpy.reshape(numpy.array(history_loss_jump), (-1,))
		self.history_loss_jump = numpy.concatenate([self.history_loss_jump, history_loss_jump])
		
		history_loss_valid = numpy.reshape(numpy.array(history_loss_valid), (-1,))
		self.history_loss_valid = numpy.concatenate([self.history_loss_valid, history_loss_valid])
		
		history_loss = numpy.reshape(numpy.array(history_loss), (-1,))
		self.history_loss = numpy.concatenate([self.history_loss, history_loss])

	def save_environment(self):

		dct = 	{
					'dense1': self.dense1.get_weights(),
					'dense2': self.dense2.get_weights(),
					'history_loss_jump': self.history_loss_jump,
					'history_loss_valid': self.history_loss_valid,
					'history_loss': self.history_loss,
					'test_data': self.test_data,
					'global_step': self.global_step.eval()
				}

		pickle.dump(dct, open(self._save_environment, "wb"))

	def load_environment(self):

		session = tensorflow.get_default_session()
		session.run(tensorflow.global_variables_initializer())

		dct = pickle.load(open(self._load_environment, "rb"))

		self.dense1.set_weights(dct['dense1'])
		self.dense2.set_weights(dct['dense2'])
		self.history_loss_jump = dct['history_loss_jump']
		self.history_loss_valid = dct['history_loss_valid']
		self.history_loss = dct['history_loss']
		self.test_data = dct['test_data']

		tensorflow.assign(self.global_step, dct['global_step']).eval()

	def plot_loss_curves(self):

		window_size = 1

		history_loss_jump = self.history_loss_jump
		history_loss_jump = [history_loss_jump[i:i+window_size].mean() for i in range(0, len(history_loss_jump), window_size)]
		plt.plot(list(range(len(history_loss_jump))), history_loss_jump, 'r', label = 'loss_jump')

		history_loss_valid = self.history_loss_valid
		history_loss_valid = [history_loss_valid[i:i+window_size].mean() for i in range(0, len(history_loss_valid), window_size)]
		plt.plot(list(range(len(history_loss_valid))), history_loss_valid, 'g', label = 'loss_valid')

		history_loss = self.history_loss
		history_loss = [history_loss[i:i+window_size].mean() for i in range(0, len(history_loss), window_size)]
		plt.plot(list(range(len(history_loss))), history_loss, 'b', label = 'loss')
		
		plt.legend()

		if (self.save_figure):
			plt.savefig(self.save_figure)
		else:
			plt.show()

	def evaluate_environment(self):

		session = tensorflow.get_default_session()
		
		number_of_walks = self.number_of_walks

		for walk_length in self.walk_lengths:

			random_walks = self.dataset.create_random_walk_routes(number_of_walks = number_of_walks, walk_length = walk_length)
			
			losses = []
			jumps = []
			distance_to_nearest_entities = []
			
			for head_id, relation_id, tail_id, target_id in tqdm(random_walks):

				target_entity_embedding_vector = self.dataset.entityid2vec[target_id:target_id+1]
				
				head_entity_embedding_vector = self.dataset.entityid2vec[head_id:head_id+1]
				relation_embedding_vector = self.dataset.relationid2vec[relation_id:relation_id+1]				
				candidate_entity_embedding_vector = self.dataset.entityid2vec[tail_id:tail_id+1]

				distance_before = numpy.linalg.norm(head_entity_embedding_vector - target_entity_embedding_vector)

				feed_dict = {
								self.head_entity_embedding_vector: head_entity_embedding_vector,
								self.relation_embedding_vector: relation_embedding_vector,
								self.target_entity_embedding_vector: target_entity_embedding_vector,
								self.candidate_entity_embedding_vector: candidate_entity_embedding_vector
							}

				predict_tail_entity_embedding_vector, loss_jump, loss_valid, loss = session.run([self.predict_tail_entity_embedding_vector, self.loss_jump, self.loss_valid, self.loss], feed_dict = feed_dict)
				losses.append([loss_jump, loss_valid, loss])

				predict_tail_entity_embedding_vector = numpy.reshape(predict_tail_entity_embedding_vector, (-1,))
				distance = list(map(lambda x: numpy.linalg.norm(self.dataset.entityid2vec[x] - predict_tail_entity_embedding_vector), range(self.dataset.entity_count)))

				distance_to_nearest_entity = min(distance)
				predict_tail_id = numpy.argmin(distance)

				distance_to_nearest_entities.append(distance_to_nearest_entity)

				distance_after = numpy.linalg.norm(predict_tail_entity_embedding_vector - target_entity_embedding_vector)

				jump = distance_before - distance_after
				jumps.append(jump)

				candidate_triplets = self.dataset.triplets[self.dataset.triplets[:, 0] == head_id]
				
				if not len(candidate_triplets):
					candidate_id = target_id
	
				else:		
					_candidate_triplets = candidate_triplets[candidate_triplets[:, 1] == relation_id]

					if len(_candidate_triplets):
						candidate_triplets = _candidate_triplets

					_candidate_ids = candidate_triplets[:, 2]
					candidate_id = min(_candidate_ids, key = lambda x: numpy.linalg.norm(self.dataset.entityid2vec[target_id] - self.dataset.entityid2vec[x]))

				distance_candidate_all_entities = numpy.linalg.norm(self.dataset.entityid2vec - self.dataset.entityid2vec[candidate_id], axis = 1)
				distance_candidate_predicted = distance_candidate_all_entities[predict_tail_id]

				rank = numpy.sort(distance_candidate_all_entities).tolist().index(distance_candidate_predicted) + 1
				# print(rank)

				self.test_data.append({
						'walk_length': walk_length,
						'head_id': head_id,
						'relation_id': relation_id,
						'tail_id': tail_id,
						'target_id': target_id,
						'loss_jump': loss_jump,
						'loss_valid': loss_valid, 
						'loss': loss,
						'jump': jump,
						'distance_to_nearest_entity': distance_to_nearest_entity,
						'predict_tail_id': predict_tail_id,
						'rank_predict_tail_id': rank
					})

			losses = numpy.array(losses)

	def summarize_results(self):

		dct = {}

		for test in self.test_data:
			if test['walk_length'] not in dct:
				dct[test['walk_length']] = {'loss_jump': [], 'loss_valid': [], 'loss': [], 'jump': [], 'distance_to_nearest_entity': [], 'rank_predict_tail_id': []}

			dct[test['walk_length']]['loss_jump'].append(test['loss_jump'])		
			dct[test['walk_length']]['loss_valid'].append(test['loss_valid'])		
			dct[test['walk_length']]['loss'].append(test['loss'])		
			dct[test['walk_length']]['jump'].append(test['jump'])		
			dct[test['walk_length']]['distance_to_nearest_entity'].append(test['distance_to_nearest_entity'])
			dct[test['walk_length']]['rank_predict_tail_id'].append(test['rank_predict_tail_id'])

		for walk_length in dct:

			dct[walk_length]['rank_predict_tail_id'] = numpy.array(dct[walk_length]['rank_predict_tail_id'])

			print("Walk Length:", walk_length)

			print("Mean Loss Jump:", numpy.mean(dct[walk_length]['loss_jump']))
			print("Mean Loss Valid:", numpy.mean(dct[walk_length]['loss_valid']))
			print("Mean Loss:", numpy.mean(dct[walk_length]['loss']))
			print("Mean Jump:", numpy.mean(dct[walk_length]['jump']))
			print("Mean Distance to nearest entity:", numpy.mean(dct[walk_length]['distance_to_nearest_entity']))
			print("Mean Rank:", numpy.mean(dct[walk_length]['rank_predict_tail_id']))
			print("Mean Reciprocal Rank:", numpy.mean(1 / dct[walk_length]['rank_predict_tail_id']))
			print("Hits@1:", len(dct[walk_length]['rank_predict_tail_id'][dct[walk_length]['rank_predict_tail_id'] <= 1])/len(dct[walk_length]['rank_predict_tail_id']))
			print("Hits@3:", len(dct[walk_length]['rank_predict_tail_id'][dct[walk_length]['rank_predict_tail_id'] <= 3])/len(dct[walk_length]['rank_predict_tail_id']))
			print("Hits@10:", len(dct[walk_length]['rank_predict_tail_id'][dct[walk_length]['rank_predict_tail_id'] <= 10])/len(dct[walk_length]['rank_predict_tail_id']))

			print()		

if __name__ == "__main__":

	with tensorflow.Session() as session:

		Environment(	
			dataset = Countries(
					load_dataset = None,
					embedding_model = TransE, 
					embedding_size = 25, 
					margin = 1.0, 
					learning_rate = 1e-3, 
					patience = 50, 
					batch_size = 128, 
					sampling_type = 'bernoulli', 
					save_dataset = "dataset_Countries"
				),						
			load_environment = None,
			train_environment = False, 
			number_of_steps = 50000,
			learning_rate = 1e-3,
			plot_loss_curves = False,
			save_figure = "lossCurves_Countries.png",
			evaluate_environment = True,
			number_of_walks = 250,
			walk_lengths = [1, 2, 3, 5],
			save_environment = "environment_Countries"		
		)