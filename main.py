import tensorflow
import numpy
import pandas
import pickle
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
from Models.TransE import TransE
from Models.TransH import TransH
from Models.TransR import TransR
from Models.TransD import TransD
from Datasets.Countries import Countries
from Datasets.Kinship import Kinship
from Datasets.UMLS import UMLS
from Datasets.FB15K import FB15K
from Datasets.NELL995 import NELL995
from Environment import Environment

parser = argparse.ArgumentParser()
parser.add_argument("-ds", "--dataset", help = "Dataset object to be used", type = str, default = None)
parser.add_argument("-ld", "--load-dataset", help = "Load pickle file of the dataset", type = str, default = None)
parser.add_argument("-em", "--embedding-model", help = "Embedding Model to be used", type = str, default = 'TransE')
parser.add_argument("-es", "--embedding-size", help = "Embedding size of each vector", type = int, default = 25)
parser.add_argument("-bs", "--batch-size", help = "Batch size while training embeddings", type = int, default = 128)
parser.add_argument("-mg", "--margin", help = "Margin of error allowed in the loss", type = float, default = 1)
parser.add_argument("-lr", "--learning-rate", help = "Learning rate for the optimizer", type = float, default = 1e-3)
parser.add_argument("-pt", "--patience", help = "Patience while training the embedding model for loss to improve", type = int, default = 50)
parser.add_argument("-st", "--sampling-type", help = "Method used to sample data", type = str, default = 'bernoulli')
parser.add_argument("-sd", "--save-dataset", help = "Pickle file name for the trained dataset for saving", type = str, default = None)
parser.add_argument("-le", "--load-environment", help = "Pickle file name for the trained environment for loading", type = str, default = None)
parser.add_argument("-te", "--train-environment", help = "Train Environment", default = False, action = 'store_true')
parser.add_argument("-ns", "--number-steps", help = "Number of steps to train environment", type = int, default = 200000)
parser.add_argument("-pc", "--plot-curves", help = "Plot loss curves", default = False, action = 'store_true')
parser.add_argument("-sf", "--save-figure", help = "Save plot loss curves to a file", type = str, default = None)
parser.add_argument("-ee", "--evaluate-environment", help = "Evaluate Environment", default = False, action = 'store_true')
parser.add_argument("-nw", "--number-walks", help = "Number of random walks to perform", type = int, default = 250)
parser.add_argument("-wl", "--walk-lengths", help = "Walk lengths", nargs = '+', type = int)
parser.add_argument("-se", "--save-environment", help = "Pickle file name for the trained environment for saving", type = str, default = None)
parser.add_argument("-sr", "--summarize-results", help = "Display summary of results", default = False, action = 'store_true')
args = parser.parse_args()

with tensorflow.Session() as session:

		Environment(	
			dataset = eval(args.dataset)(
					load_dataset = args.load_dataset,
					embedding_model = eval(args.embedding_model), 
					embedding_size = args.embedding_size, 
					margin = args.margin, 
					learning_rate = args.learning_rate, 
					patience = args.patience, 
					batch_size = args.batch_size, 
					sampling_type = args.sampling_type, 
					save_dataset = args.save_dataset
				),						
			load_environment = args.load_environment,
			train_environment = args.train_environment, 
			number_of_steps = args.number_steps,
			learning_rate = args.learning_rate,
			plot_loss_curves = args.plot_curves,
			save_figure = args.save_figure,
			evaluate_environment = args.evaluate_environment,
			number_of_walks = args.number_walks,
			walk_lengths = args.walk_lengths,
			save_environment = args.save_environment,
			summarize_results = args.summarize_results
		)