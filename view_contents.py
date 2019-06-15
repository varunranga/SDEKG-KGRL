import argparse
import pickle
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument("-fn", "--filename", help = "Filename of the file to view its contents", type = str, default = None)
args = parser.parse_args()

pprint(pickle.load(open(args.filename, "rb")))
