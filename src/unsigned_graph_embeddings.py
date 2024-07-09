import glob
import networkx as nx
from karateclub import Graph2Vec, BoostNE, Walklets, FGSD, DeepWalk, GraphWave, SF
from node2vec import Node2Vec
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import SVC

'''
Parse the configuration file. Each line corresponds to a method.

file: File to parse.
Return: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
'''
def parse(file):
	params = {}
	f = open(file, "r")
	for line in f:
		# separator = tab
		line = line.split("	")
		params[line[0].lower()] = eval(line[1])
	return params

'''
Loads graphs.

path: path to the graphml files.
Return: List of networkx graphs and associated labels.
'''
def load_graphs(path):
	graphs = []
	labels = []
	filelist = glob.glob(path+"/*.graphml")

	for i in range(len(filelist)):
		file = path+"/%s.graphml" % i
		g = nx.read_graphml(file)
		g = nx.convert_node_labels_to_integers(g)
		graphs.append(g)
		labels.append(int(g.graph["label"]))
	return graphs, labels

def compute_res(X, y):

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


	model = SVC(class_weight='balanced', probability=True)
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	print ("Micro F-measure: %0.4f" % (f1_score(y_test, y_pred, average='micro')))
	print ("Macro F-measure: %0.4f" % (f1_score(y_test, y_pred, average='macro')))
	print ("Binary F-measure: %0.4f" % (f1_score(y_test, y_pred, average='binary', pos_label=1)))

'''
Generates and saves graph2vec embeddings in a file.
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.

Return: List of embeddings (1 per graph in G)
'''
def gen_Graph2vec(G, parameters):

	try:
		params = parameters["graph2vec"]
	except:
		params = {}

	model = Graph2Vec(**params)
	model.fit(G)
	X = model.get_embedding()

	return X


def gen_BoostNE(G, parameters):
	embeddings = []

	try:
		params = parameters["boostne"]
	except:
		params = {}

	for g in G:
		target_vertex_uid = int(g.graph['target_uid'])
		target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
		target_vertex = int(target_vertex[0])
		model = BoostNE(**params)
		model.fit(g)
		X = model.get_embedding()

		for vertex_id in range(len(X)):
			g_embedding = [0]*len(X[0])
			#save embedding of the targeted node
			if vertex_id == target_vertex:
				embeddings.append(list(X[vertex_id]))

	return embeddings

def gen_Walklets(G, parameters):
	embeddings = []

	try:
		params = parameters["walklets"]
	except:
		params = {}

	for g in G:
		target_vertex_uid = int(g.graph['target_uid'])
		target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
		target_vertex = int(target_vertex[0])
		model = Walklets(**params)
		model.fit(g)
		X = model.get_embedding()

		for vertex_id in range(len(X)):
			g_embedding = [0]*len(X[0])
			#save embedding of the targeted node
			if vertex_id == target_vertex:
				embeddings.append(list(X[vertex_id]))

	return embeddings

def gen_DeepWalk(G, parameters):
	embeddings = []

	try:
		params = parameters["deepwalk"]
	except:
		params = {}

	for g in G:
		target_vertex_uid = int(g.graph['target_uid'])
		target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
		target_vertex = int(target_vertex[0])
		model = DeepWalk(**params)
		model.fit(g)
		X = model.get_embedding()

		for vertex_id in range(len(X)):
			g_embedding = [0]*len(X[0])
			#save embedding of the targeted node
			if vertex_id == target_vertex:
				embeddings.append(list(X[vertex_id]))

	return embeddings

def gen_GraphWave(G, parameters):
	embeddings = []

	try:
		params = parameters["graphwave"]
	except:
		params = {}

	for g in G:
		target_vertex_uid = int(g.graph['target_uid'])
		target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
		target_vertex = int(target_vertex[0])
		model = GraphWave(**params)
		model.fit(g)
		X = model.get_embedding()

		for vertex_id in range(len(X)):
			g_embedding = [0]*len(X[0])
			#save embedding of the targeted node
			if vertex_id == target_vertex:
				embeddings.append(list(X[vertex_id]))

	return embeddings

def gen_FGSD(G, parameters):

	try:
		params = parameters["fgsd"]
	except:
		params = {}

	#fsgd works with undirected graphs
	for i in range(len(G)):
		G[i] = G[i].to_undirected()

	
	model = FGSD(**params)
	model.fit(G)
	X = model.get_embedding()

	return X

def gen_SF(G, parameters):

	try:
		params = parameters["sf"]
	except:
		params = {}

	model = SF(**params)
	model.fit(G)
	X = model.get_embedding()

	return X

def gen_Node2vec(G, parameters):
	embeddings = []
	try:
		params = parameters["node2vec"]
	except:
		params = {}

	for g in G:
		target_vertex_uid = int(g.graph['target_uid'])
		target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
		target_vertex = int(target_vertex[0])
		model = Node2Vec(g, **params)
		model = model.fit(window=10, min_count=1)

		for vertex_id in range(len(g)):
			X = model.wv[str(vertex_id)]
			g_embedding = [0]*len(X)
			#save embedding of the targeted node
			if vertex_id == target_vertex:
				embeddings.append(list(X))

	return embeddings



if __name__ == "__main__":

	p = argparse.ArgumentParser()
	p.add_argument("--graph2vec", action="store_true")
	p.add_argument("--fgsd", action="store_true")
	p.add_argument("--sf", action="store_true")
	p.add_argument("--node2vec", action="store_true")
	p.add_argument("--boostne", action="store_true")
	p.add_argument("--walklets", action="store_true")
	p.add_argument("--deepwalk", action="store_true")
	p.add_argument("--graphwave", action="store_true")
	p.add_argument("--camembert", action="store_true")
	p.add_argument("--flaubert", action="store_true")
	p.add_argument("--combine", action="store_true")
	args = p.parse_args()

	params = parse("src/config.txt")
	G, labels = load_graphs("in/graphs/SpaceOrigin_graphs/signed")


	features = []

	if args.graph2vec:
		X = gen_Graph2vec(G, params)
		print ("Graph2vec")
		compute_res(X, labels)

	if args.fgsd:
		X = gen_FGSD(G, params)
		if args.combine:
			features.append(X)
		else:
			X = split(X)
			print ("FGSD")
			compute_CV(X, labels)

	if args.sf:
		X = gen_SF(G, params)
		print ("SF")
		compute_res(X, labels)

	if args.boostne:
		X = gen_BoostNE(G, params)
		if args.combine:
			features.append(X)
		else:
			print ("BoostNE")
			compute_res(X, labels)
	
	if args.walklets:
		X = gen_Walklets(G, params)
		if args.combine:
			features.append(X)
		else:
			print ("Walklets")
			compute_res(X, labels)

	if args.deepwalk:
		X = gen_DeepWalk(G, params)
		if args.combine:
			features.append(X)
		else:
			print ("Deepwalk")
			compute_res(X, labels)

	if args.graphwave:
		X = gen_GraphWave(G, params)
		if args.combine:
			features.append(X)
		else:
			print ("GraphWave")
			compute_res(X, labels)
	
	if args.node2vec:
		X  = gen_Node2vec(G, params)
		if args.combine:
			features.append(X)
		else:
			print ("Node2vec")
			compute_res(X, labels)
	
	if args.camembert:
		camembert = camembert_features()
		if args.combine:
			features.append(camembert)
		else:
			print ("Camembert")
			compute_CV(camembert, labels)

	if args.flaubert:
		camembert = flaubert_features()
		if args.combine:
			features.append(flaubert)
		else:
			print ("Flaubert")
			compute_CV(flaubert, labels)

	if args.combine:
		X = features[0]
		for i in range(len(features)-1):
			X = np.concatenate((X, features[i+1]), axis=1)
		
		compute_CV(X, labels)
	