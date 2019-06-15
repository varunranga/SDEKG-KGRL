import pandas
import numpy
import os
import pickle
import tensorflow
from tqdm import tqdm
from math import inf
from math import ceil
from Models.TransE import TransE
from Models.TransH import TransH
from Models.TransR import TransR
from Models.TransD import TransD

class NELL995():

	def __init__(self, load_dataset = None, embedding_model = None, embedding_size = None, margin = None, learning_rate = None, patience = None, batch_size = None, sampling_type = None, save_dataset = None):

		if not load_dataset:

			self.__name__ = "NELL995"
			self.embedding_size = embedding_size
			self.margin = margin
			self.learning_rate = learning_rate
			self.patience = patience
			self.batch_size = batch_size
			self.sampling_type = sampling_type
			self.embedding_model = embedding_model

			df = pandas.read_csv("./Datasets/NELL995/raw.kb", header = -1, sep = "\t")
			triplets = df.as_matrix()

			entities = sorted(list(set(triplets[:, 0].tolist() + triplets[:, 2].tolist())))
			relations = sorted(list(set(triplets[:, 1].tolist())))

			self.entity_count = len(entities)
			self.relation_count = len(relations)

			self.entity2id = {entity:i for i, entity in enumerate(entities)}
			self.id2entity = {i:entity for i, entity in enumerate(entities)}

			self.relation2id = {relation:i for i, relation in enumerate(relations)}
			self.id2relation = {i:relation for i, relation in enumerate(relations)}

			self.triplets = numpy.array([[self.entity2id[h], self.relation2id[r], self.entity2id[t]] for h, r, t in triplets])

			self.bernoulli_probablities = {}
			for i in range(self.relation_count):
				data = self.triplets[self.triplets[:, 1] == i]
				head_entities = set(data[:, 0])
				tph = sum(len(data[data[:, 0] == head_entity]) for head_entity in head_entities) / len(head_entities)
				tail_entities = set(data[:, 1])
				hpt = sum(len(data[data[:, 1] == tail_entity]) for tail_entity in tail_entities) / len(tail_entities)
				self.bernoulli_probablities[i] = tph / (hpt + tph)

			self.entityid2vec, self.relationid2vec = self.create_knowledge_graph_embeddings()

			if save_dataset:

				dct = 	{
							'__name__': self.__name__,
							'entity_count': self.entity_count,
							'relation_count': self.relation_count,
							'embedding_model': self.embedding_model,
							'embedding_size': self.embedding_size,
							'margin': self.margin,
							'learning_rate': self.learning_rate,
							'patience': self.patience,
							'batch_size': self.batch_size,
							'sampling_type': self.sampling_type,
							'entity2id': self.entity2id,
							'relation2id': self.relation2id,
							'triplets': self.triplets,
							'bernoulli_probablities': self.bernoulli_probablities,
							'entityid2vec': self.entityid2vec,
							'relationid2vec': self.relationid2vec
						}

				pickle.dump(dct, open(save_dataset, "wb"))		

		else:

			dct = pickle.load(open(load_dataset, "rb"))

			self.__name__ = dct['__name__']
			self.embedding_model = dct['embedding_model']
			self.margin = dct['margin']
			self.learning_rate = dct['learning_rate']
			self.patience = dct['patience']
			self.batch_size = dct['batch_size']
			self.sampling_type = dct['sampling_type']
			self.entity_count = dct['entity_count']
			self.relation_count = dct['relation_count']
			self.embedding_size = dct['embedding_size']
			self.entity2id = dct['entity2id']
			self.relation2id = dct['relation2id']
			self.triplets = dct['triplets']
			self.bernoulli_probablities = dct['bernoulli_probablities']
			self.entityid2vec = dct['entityid2vec']
			self.relationid2vec = dct['relationid2vec']

	def create_knowledge_graph_embeddings(self):

		session = tensorflow.get_default_session()

		config = 	{
						'entity_count': self.entity_count,
						'relation_count': self.relation_count,
						'embedding_size': self.embedding_size,
						'margin': self.margin,
						'learning_rate': self.learning_rate
					}

		embedding_model = self.embedding_model(config)

		self.embedding_model = embedding_model.__name__	

		waited = 0
		best_loss = inf
		epoch = 0

		session.run(tensorflow.global_variables_initializer())

		while waited < self.patience:
			print()
			print("EPOCH", epoch + 1)
			
			train_generator = self.create_train_data_generator().__iter__()
			
			training_step = 0
			training_loss = 0
			
			for _  in tqdm(range(ceil(len(self.triplets) / self.batch_size))):
				positive_heads, positive_tails, positive_relations, negative_heads, negative_tails, negative_relations = train_generator.__next__()
			
				training_loss += embedding_model.train_model(session, positive_heads, positive_tails, positive_relations, negative_heads, negative_tails, negative_relations)
				training_step += 1

			average_training_loss = training_loss / training_step
			print("Train loss:", average_training_loss)
			
			if average_training_loss < best_loss:
				best_loss = average_training_loss
				waited = 0

				best_entity_embeddings = embedding_model.entity_embedding_vectors.eval()
				best_relation_embeddings = embedding_model.relation_embedding_vectors.eval()

			else:
				waited += 1

			epoch += 1

			print("-"*79)
			print()

		return best_entity_embeddings, best_relation_embeddings

	def create_train_data_generator(self):

		data = self.triplets
		batch_size = self.batch_size
		sampling_type = self.sampling_type

		if sampling_type == 'uniform':
			
			for i in range(0, len(data), batch_size):
				positive_heads = data[i:i+batch_size, 0]
				positive_tails = data[i:i+batch_size, 2]
				positive_relations = data[i:i+batch_size, 1]
				
				negative_heads = positive_heads.copy()
				negative_relations = positive_relations.copy()
				negative_tails = positive_tails.copy()

				for j in range(len(positive_heads)):
					if numpy.random.rand() < 0.5:
						negative_heads[j] = numpy.random.randint(0, self.entity_count)
					else:
						negative_tails[j] = numpy.random.randint(0, self.entity_count)

				yield positive_heads, positive_tails, positive_relations, negative_heads, negative_tails, negative_relations

		elif sampling_type == 'bernoulli':
			
			for i in range(0, len(data), batch_size):
				positive_heads = data[i:i+batch_size, 0]
				positive_tails = data[i:i+batch_size, 2]
				positive_relations = data[i:i+batch_size, 1]

				negative_heads = positive_heads.copy()
				negative_relations = positive_relations.copy()
				negative_tails = positive_tails.copy()

				for j in range(len(positive_heads)):

					if numpy.random.rand() < self.bernoulli_probablities[negative_relations[j]]:
						negative_heads[j] = numpy.random.randint(0, self.entity_count)
					else:
						negative_tails[j] = numpy.random.randint(0, self.entity_count)

						
				yield positive_heads, positive_tails, positive_relations, negative_heads, negative_tails, negative_relations

	def create_random_walk_routes(self, number_of_walks, walk_length):

		walks = []
		walks_done = 0

		source_entities = list(set(self.triplets[:, 0].tolist()))

		while walks_done < number_of_walks:

			source_id = numpy.random.choice(source_entities)

			walk = []
			_walk_length = 0

			while _walk_length < walk_length:

				candidate_triplets = self.triplets[self.triplets[:, 0] == source_id]
				
				if len(candidate_triplets) == 0:
					break

				random_triplet = candidate_triplets[numpy.random.randint(0, len(candidate_triplets))]

				walk.append(random_triplet)
				source_id = random_triplet[2]
				_walk_length += 1		

			if _walk_length < walk_length:
				continue
		
			walk = [walk[0][0], walk[0][1], walk[0][2], walk[-1][-1]]

			walks.append(numpy.array(walk))
			walks_done += 1

		walks = numpy.array(walks)

		return walks


if __name__ == "__main__":

	with tensorflow.Session() as session:

		NELL995(
			load_dataset = False,
			embedding_model = TransE, 
			embedding_size = 25, 
			margin = 1.0, 
			learning_rate = 1e-3, 
			patience = 50, 
			batch_size = 128, 
			sampling_type = 'bernoulli', 
			save_dataset = "Dataset_NELL995_MAIN"
		)
