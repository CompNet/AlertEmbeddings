from src.text_embeddings import create_text_embeddings
from src.unsigned_graph_embeddings import *
import os
import glob
import csv


if __name__ == '__main__':
	#run text_embeddings
	create_text_embeddings()

	#run graph embeddings
	os.system("python src/unsigned_graph_embeddings.py --deepwalk")

	#run SG2V
	os.system('python signed_graph2vec/graph2vec.py --input-path "in/graphs/SpaceOrigin_graphs/signed/*.graphml" --output-path emb/sg2v_embeddings.csv')


	#run SGCN
	for i in range(len(glob.glob("in/graphs/SpaceOrigin_graphs/signed/*.graphml"))):
		file = "in/graphs/SpaceOrigin_graphs/signed/%s.graphml" % i
		g = nx.read_graphml(file)
		with open('in/tmp/edges.csv', 'w') as file:
			fieldnames = ['Node id 1', 'Node id 2', 'Sign']
			writer = csv.DictWriter(file, fieldnames=fieldnames)
			writer.writeheader()
			for (n1, n2, d) in g.edges(data=True):
				if d["sign"] == "+":
					s = 1
				if d["sign"] == "-":
					s = 0
				writer.writerow({'Node id 1': n1.replace("n", ""), 'Node id 2': n2.replace("n", ""), 'Sign': s})
		os.system('python SGCN/main.py --layers 96 32 --learning-rate 0.01 --reduction-dimensions 64 --epochs 10 --reduction-iterations 10 --edge-path in/tmp/edges.csv --embedding-path emb/sgcn/%s.csv --regression-weights-path output/weights.csv') % (i)