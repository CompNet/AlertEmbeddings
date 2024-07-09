from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from karateclub import Graph2Vec, FGSD, SF, GL2Vec, NetLSD
from karateclub import BoostNE, Walklets, GraRep, DeepWalk, NMFADMM, GraphWave, Role2Vec, NodeSketch, NetMF, Role2Vec
from node2vec import Node2Vec

import numpy as np
import networkx as nx
import glob
import time
import argparse
import csv

'''
Split a list in 10 folds following the splitting strategy used in https://www.frontiersin.org/articles/10.3389/fdata.2019.00008/full.
X: List to split

Return: List of 10 parts for the cross validation. Each part is a tuple (train, test) with a 70/30% repartition.
'''
def split(X):
	if len(X) == 1320:
		#Corpus equilibre
		splits = [X[:137], X[137:258], X[258:395], X[395:516], X[516:639], X[639:774], X[774:899], X[899:1044], X[1044:1171], X[1171:1320]]
	if len(X) == 2545:
		#Corpus complet
		splits = [X[:258], X[258:509], X[509:764], X[764:1014], X[1014:1266], X[1266:1520], X[1520:1769], X[1769:2032], X[2032:2279], X[2279:2545]]

	ret = []
	for i in range(10):
		test_splits = [v % 10 for v in range(i, i+3)]
		train_splits = [v % 10 for v in range(i+3, i+10)]
		train, test = [], []
		for k in train_splits:
			train.extend(splits[k])
		for k in test_splits:
			test.extend(splits[k])
		split = (train, test)
		ret.append(split)
		'''test_splits = [v % 10 for v in range(i, i+3)]
		test_splits.sort()
		train_splits = [v % 10 for v in range(i+3, i+10)]
		train_splits.sort()
		train, test = [], []
		for k in range(10):
			if k in train_splits:
				train.extend(splits[k])
			if k in test_splits:
				test.extend(splits[k])
		split = (train, test)
		ret.append(split)'''

	return ret

'''
Generates and saves graph2vec embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: List of embeddings (1 per graph in G)
'''
def gen_Graph2vec(G_train, G_test, parameters, regenerate=False, timer=True):

	G = G_train + G_test
	try:
		params = parameters["graph2vec"]
	except:
		params = {}

	start_time = time.time()
	filename = "Graph2vec_WP" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			X = np.load('emb/Graph2vec/%s.npy' % filename)
		except:
			model = Graph2Vec(**params)
			model.fit(G)
			X = model.get_embedding()
			np.save('emb/Graph2vec/%s.npy' % filename, X)
	else:
		model = Graph2Vec(**params)
		model.fit(G)
		X = model.get_embedding()
		np.save('emb/Graph2vec/%s.npy' % filename, X)

	if timer:
		print(" == Graph2vec Runtime : %s s ==" % (time.time() - start_time))

	X_train = X[:len(G_train)]
	X_test = X[len(G_train):]
	return X_train, X_test

'''
Creates a list of graphs from graphml files.

corpus: "complet" or "equilibre", the corpus to load.
gtype: "full", "before" or "after"
Return: List of networkx graphs.
'''
def load_graphs(corpus, gtype):
	graphs = []

	for i in range(2545):
		file = "graphs/signed/%s.graphml" % i
		g = nx.read_graphml(file)
		g = nx.convert_node_labels_to_integers(g)
		graphs.append(g)
	return graphs

'''
Computes a 10 fold cross validation.

data: List of 10 tuples (train, test) [features]
labels: List of 10 tuples (train, test) [labels]
'''
def compute_res(X_train, X_test, y_train, y_test):

	scaler = StandardScaler().fit(X_train)
	train = scaler.transform(X_train)
	test = scaler.transform(X_test)

	model = SVC(class_weight='balanced', probability=True)
	model.fit(train, y_train)
	y_pred = model.predict(test)

	print ("Micro F-measure: %0.4f" % (f1_score(y_test, y_pred, average='micro')))
	print ("Macro F-measure: %0.4f" % (f1_score(y_test, y_pred, average='macro')))
	print ("Binary F-measure: %0.4f" % (f1_score(y_test, y_pred, average='binary', pos_label=1)))

	

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
Generates and saves BoostNE embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: Embedding of the targeted node and general embedding generated by averaging embeddings of all nodes in the graph.
'''
def gen_BoostNE(G, parameters, regenerate=False, timer=True):
	embeddings = []
	graph_embeddings = []
	try:
		params = parameters["boostne"]
	except:
		params = {}

	start_time = time.time()
	filename = "BoostNE" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			embeddings = np.load('emb/BoostNE/node_%s.npy' % filename)
			graph_embeddings = np.load('emb/BoostNE/graph_%s.npy' % filename)
		except:
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

					#global graph embedding (average of all nodes)
					for f_id in range(len(X[vertex_id])):
						g_embedding[f_id] += float(X[vertex_id][f_id])
				g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
				graph_embeddings.append(g_emb)

	else:
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

				#global graph embedding (average of all nodes)
				for f_id in range(len(X[vertex_id])):
					g_embedding[f_id] += float(X[vertex_id][f_id])
			g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
			graph_embeddings.append(g_emb)

	if timer:
		print(" == BoostNE Runtime : %s s ==" % (time.time() - start_time))

	np.save('emb/BoostNE/node_%s' % filename, embeddings)
	np.save('emb/BoostNE/graph_%s' % filename, graph_embeddings)
	return embeddings, graph_embeddings

'''
Generates and saves Walklets embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: Embedding of the targeted node and general embedding generated by averaging embeddings of all nodes in the graph.
'''
def gen_Walklets(G, parameters, regenerate=False, timer=True):
	embeddings = []
	graph_embeddings = []
	try:
		params = parameters["walklets"]
	except:
		params = {}

	start_time = time.time()
	filename = "Walklets" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			embeddings = np.load('emb/Walklets/node_%s.npy' % filename)
			graph_embeddings = np.load('emb/Walklets/graph_%s.npy' % filename)
		except:
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

					#global graph embedding (average of all nodes)
					for f_id in range(len(X[vertex_id])):
						g_embedding[f_id] += float(X[vertex_id][f_id])
				g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
				graph_embeddings.append(g_emb)

	else:
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

				#global graph embedding (average of all nodes)
				for f_id in range(len(X[vertex_id])):
					g_embedding[f_id] += float(X[vertex_id][f_id])
			g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
			graph_embeddings.append(g_emb)
	if timer:
		print(" == Walklets Runtime : %s s ==" % (time.time() - start_time))
	np.save('emb/Walklets/node_%s.npy' % filename, embeddings)
	np.save('emb/Walklets/graph_%s.npy' % filename, graph_embeddings)
	return embeddings, graph_embeddings

'''
Generates and saves DeepWalk embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: Embedding of the targeted node and general embedding generated by averaging embeddings of all nodes in the graph.
'''
def gen_DeepWalk(G, parameters, regenerate=False, timer=True):
	embeddings = []
	graph_embeddings = []
	try:
		params = parameters["deepwalk"]
	except:
		params = {}

	start_time = time.time()
	filename = "DeepWalk" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			embeddings = np.load('emb/DeepWalk/node_%s.npy' % filename)
			graph_embeddings = np.load('emb/DeepWalk/graph_%s.npy' % filename)
		except:
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

					#global graph embedding (average of all nodes)
					for f_id in range(len(X[vertex_id])):
						g_embedding[f_id] += float(X[vertex_id][f_id])
				g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
				graph_embeddings.append(g_emb)

	else:
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

				#global graph embedding (average of all nodes)
				for f_id in range(len(X[vertex_id])):
					g_embedding[f_id] += float(X[vertex_id][f_id])
			g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
			graph_embeddings.append(g_emb)

	if timer:
		print(" == DeepWalk Runtime : %s s ==" % (time.time() - start_time))
	np.save('emb/DeepWalk/node_%s.npy' % filename, embeddings)
	np.save('emb/DeepWalk/graph_%s.npy' % filename, graph_embeddings)
	return embeddings, graph_embeddings

'''
Generates and saves NMFADMM embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: Embedding of the targeted node and general embedding generated by averaging embeddings of all nodes in the graph.
'''
def gen_NMFADMM(G, parameters, regenerate=False, timer=True):
	embeddings = []
	graph_embeddings = []
	try:
		params = parameters["nmfadmm"]
	except:
		params = {}

	start_time = time.time()
	filename = "NMFADMM" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			embeddings = np.load('emb/NMFADMM/node_%s.npy' % filename)
			graph_embeddings = np.load('emb/NMFADMM/graph_%s.npy' % filename)
		except:
			for g in G:
				target_vertex_uid = int(g.graph['target_uid'])
				target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
				target_vertex = int(target_vertex[0])
				model = NMFADMM(**params)
				model.fit(g)
				X = model.get_embedding()

				for vertex_id in range(len(X)):
					g_embedding = [0]*len(X[0])
					#save embedding of the targeted node
					if vertex_id == target_vertex:
						embeddings.append(list(X[vertex_id]))

					#global graph embedding (average of all nodes)
					for f_id in range(len(X[vertex_id])):
						g_embedding[f_id] += float(X[vertex_id][f_id])
				g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
				graph_embeddings.append(g_emb)

	else:
		for g in G:
			target_vertex_uid = int(g.graph['target_uid'])
			target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
			target_vertex = int(target_vertex[0])
			model = NMFADMM(**params)
			model.fit(g)
			X = model.get_embedding()

			for vertex_id in range(len(X)):
				g_embedding = [0]*len(X[0])
				#save embedding of the targeted node
				if vertex_id == target_vertex:
					embeddings.append(list(X[vertex_id]))

				#global graph embedding (average of all nodes)
				for f_id in range(len(X[vertex_id])):
					g_embedding[f_id] += float(X[vertex_id][f_id])
			g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
			graph_embeddings.append(g_emb)

	if timer:
		print(" == NMFADMM Runtime : %s s ==" % (time.time() - start_time))
	np.save('emb/NMFADMM/node_%s.npy' % filename, embeddings)
	np.save('emb/NMFADMM/graph_%s.npy' % filename, graph_embeddings)
	return embeddings, graph_embeddings

'''
Generates and saves GraRep embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: Embedding of the targeted node and general embedding generated by averaging embeddings of all nodes in the graph.
'''
def gen_GraRep(G, parameters, regenerate=False, timer=True):
	embeddings = []
	graph_embeddings = []
	try:
		params = parameters["grarep"]
	except:
		params = {}

	start_time = time.time()
	filename = "GraRep" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			embeddings = np.load('emb/GraRep/node_%s.npy' % filename)
			graph_embeddings = np.load('emb/GraRep/graph_%s.npy' % filename)
		except:
			for g in G:
				target_vertex_uid = int(g.graph['target_uid'])
				target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
				target_vertex = int(target_vertex[0])
				model = GraRep(**params)
				model.fit(g)
				X = model.get_embedding()

				for vertex_id in range(len(X)):
					g_embedding = [0]*len(X[0])
					#save embedding of the targeted node
					if vertex_id == target_vertex:
						embeddings.append(list(X[vertex_id]))

					#global graph embedding (average of all nodes)
					for f_id in range(len(X[vertex_id])):
						g_embedding[f_id] += float(X[vertex_id][f_id])
				g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
				graph_embeddings.append(g_emb)

	else:
		for g in G:
			target_vertex_uid = int(g.graph['target_uid'])
			target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
			target_vertex = int(target_vertex[0])
			model = GraRep(**params)
			model.fit(g)
			X = model.get_embedding()

			for vertex_id in range(len(X)):
				g_embedding = [0]*len(X[0])
				#save embedding of the targeted node
				if vertex_id == target_vertex:
					embeddings.append(list(X[vertex_id]))

				#global graph embedding (average of all nodes)
				for f_id in range(len(X[vertex_id])):
					g_embedding[f_id] += float(X[vertex_id][f_id])
			g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
			graph_embeddings.append(g_emb)

	if timer:
		print(" == GraRep Runtime : %s s ==" % (time.time() - start_time))
	np.save('emb/GraRep/node_%s.npy' % filename, embeddings)
	np.save('emb/GraRep/graph_%s.npy' % filename, graph_embeddings)
	return embeddings, graph_embeddings

'''
Generates and saves NodeSketch embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: Embedding of the targeted node and general embedding generated by averaging embeddings of all nodes in the graph.
'''
def gen_NodeSketch(G, parameters, regenerate=False, timer=True):
	embeddings = []
	graph_embeddings = []
	try:
		params = parameters["nodesketch"]
	except:
		params = {}

	start_time = time.time()
	filename = "NodeSketch" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			embeddings = np.load('emb/NodeSketch/node_%s.npy' % filename)
			graph_embeddings = np.load('emb/NodeSketch/graph_%s.npy' % filename)
		except:
			for g in G:
				target_vertex_uid = int(g.graph['target_uid'])
				target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
				target_vertex = int(target_vertex[0])
				model = NodeSketch(**params)
				model.fit(g)
				X = model.get_embedding()

				for vertex_id in range(len(X)):
					g_embedding = [0]*len(X[0])
					#save embedding of the targeted node
					if vertex_id == target_vertex:
						embeddings.append(list(X[vertex_id]))

					#global graph embedding (average of all nodes)
					for f_id in range(len(X[vertex_id])):
						g_embedding[f_id] += float(X[vertex_id][f_id])
				g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
				graph_embeddings.append(g_emb)

	else:
		for g in G:
			target_vertex_uid = int(g.graph['target_uid'])
			target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
			target_vertex = int(target_vertex[0])
			model = NodeSketch(**params)
			model.fit(g)
			X = model.get_embedding()

			for vertex_id in range(len(X)):
				g_embedding = [0]*len(X[0])
				#save embedding of the targeted node
				if vertex_id == target_vertex:
					embeddings.append(list(X[vertex_id]))

				#global graph embedding (average of all nodes)
				for f_id in range(len(X[vertex_id])):
					g_embedding[f_id] += float(X[vertex_id][f_id])
			g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
			graph_embeddings.append(g_emb)

	if timer:
		print(" == NodeSketch Runtime : %s s ==" % (time.time() - start_time))
	np.save('emb/NodeSketch/node_%s.npy' % filename, embeddings)
	np.save('emb/NodeSketch/graph_%s.npy' % filename, graph_embeddings)
	return embeddings, graph_embeddings

'''
Generates and saves NetMF embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: Embedding of the targeted node and general embedding generated by averaging embeddings of all nodes in the graph.
'''
def gen_NetMF(G, parameters, regenerate=False, timer=True):
	embeddings = []
	graph_embeddings = []
	try:
		params = parameters["netmf"]
	except:
		params = {}

	start_time = time.time()
	filename = "NetMF" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			embeddings = np.load('emb/NetMF/node_%s.npy' % filename)
			graph_embeddings = np.load('emb/NetMF/graph_%s.npy' % filename)
		except:
			for g in G:
				target_vertex_uid = int(g.graph['target_uid'])
				target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
				target_vertex = int(target_vertex[0])
				model = NetMF(**params)
				model.fit(g)
				X = model.get_embedding()

				for vertex_id in range(len(X)):
					g_embedding = [0]*len(X[0])
					#save embedding of the targeted node
					if vertex_id == target_vertex:
						embeddings.append(list(X[vertex_id]))

					#global graph embedding (average of all nodes)
					for f_id in range(len(X[vertex_id])):
						g_embedding[f_id] += float(X[vertex_id][f_id])
				g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
				graph_embeddings.append(g_emb)

	else:
		for g in G:
			target_vertex_uid = int(g.graph['target_uid'])
			target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
			target_vertex = int(target_vertex[0])
			model = NetMF(**params)
			model.fit(g)
			X = model.get_embedding()

			for vertex_id in range(len(X)):
				g_embedding = [0]*len(X[0])
				#save embedding of the targeted node
				if vertex_id == target_vertex:
					embeddings.append(list(X[vertex_id]))

				#global graph embedding (average of all nodes)
				for f_id in range(len(X[vertex_id])):
					g_embedding[f_id] += float(X[vertex_id][f_id])
			g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
			graph_embeddings.append(g_emb)

	if timer:
		print(" == NetMF Runtime : %s s ==" % (time.time() - start_time))
	np.save('emb/NetMF/node_%s.npy' % filename, embeddings)
	np.save('emb/NetMF/graph_%s.npy' % filename, graph_embeddings)
	return embeddings, graph_embeddings

'''
Generates and saves Role2Vec embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: Embedding of the targeted node and general embedding generated by averaging embeddings of all nodes in the graph.
'''
def gen_Role2Vec(G, parameters, regenerate=False, timer=True):
	embeddings = []
	graph_embeddings = []
	try:
		params = parameters["role2vec"]
	except:
		params = {}

	start_time = time.time()
	filename = "Role2Vec" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			embeddings = np.load('emb/Role2Vec/node_%s.npy' % filename)
			graph_embeddings = np.load('emb/Role2Vec/graph_%s.npy' % filename)
		except:
			for g in G:
				target_vertex_uid = int(g.graph['target_uid'])
				target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
				target_vertex = int(target_vertex[0])
				model = Role2Vec(**params)
				model.fit(g)
				X = model.get_embedding()

				for vertex_id in range(len(X)):
					g_embedding = [0]*len(X[0])
					#save embedding of the targeted node
					if vertex_id == target_vertex:
						embeddings.append(list(X[vertex_id]))

					#global graph embedding (average of all nodes)
					for f_id in range(len(X[vertex_id])):
						g_embedding[f_id] += float(X[vertex_id][f_id])
				g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
				graph_embeddings.append(g_emb)

	else:
		for g in G:
			target_vertex_uid = int(g.graph['target_uid'])
			target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
			target_vertex = int(target_vertex[0])
			model = Role2Vec(**params)
			model.fit(g)
			X = model.get_embedding()

			for vertex_id in range(len(X)):
				g_embedding = [0]*len(X[0])
				#save embedding of the targeted node
				if vertex_id == target_vertex:
					embeddings.append(list(X[vertex_id]))

				#global graph embedding (average of all nodes)
				for f_id in range(len(X[vertex_id])):
					g_embedding[f_id] += float(X[vertex_id][f_id])
			g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
			graph_embeddings.append(g_emb)

	if timer:
		print(" == Role2Vec Runtime : %s s ==" % (time.time() - start_time))
	np.save('emb/Role2Vec/node_%s.npy' % filename, embeddings)
	np.save('emb/Role2Vec/graph_%s.npy' % filename, graph_embeddings)
	return embeddings, graph_embeddings

'''
Generates and saves GraphWave embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: Embedding of the targeted node and general embedding generated by averaging embeddings of all nodes in the graph.
'''
def gen_GraphWave(G, parameters, regenerate=False, timer=True):
	embeddings = []
	graph_embeddings = []
	try:
		params = parameters["graphwave"]
	except:
		params = {}

	start_time = time.time()
	filename = "GraphWave" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			embeddings = np.load('emb/GraphWave/node_%s.npy' % filename)
			graph_embeddings = np.load('emb/GraphWave/graph_%s.npy' % filename)
		except:
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

					#global graph embedding (average of all nodes)
					for f_id in range(len(X[vertex_id])):
						g_embedding[f_id] += float(X[vertex_id][f_id])
				g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
				graph_embeddings.append(g_emb)

	else:
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

				#global graph embedding (average of all nodes)
				for f_id in range(len(X[vertex_id])):
					g_embedding[f_id] += float(X[vertex_id][f_id])
			g_emb = [g_embedding[x]/float(len(X[vertex_id])) for x in range(len(g_embedding))]
			graph_embeddings.append(g_emb)

	if timer:
		print(" == GraphWave Runtime : %s s ==" % (time.time() - start_time))
	np.save('emb/GraphWave/node_%s.npy' % filename, embeddings)
	np.save('emb/GraphWave/graph_%s.npy' % filename, graph_embeddings)
	return embeddings, graph_embeddings

'''
Generates and saves Node2vec embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: Embedding of the targeted node and general embedding generated by averaging embeddings of all nodes in the graph.
'''
def gen_Node2vec(G, parameters, regenerate=False, timer=True):
	embeddings = []
	graph_embeddings = []
	try:
		params = parameters["node2vec"]
	except:
		params = {}

	start_time = time.time()
	filename = "Node2vec" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			embeddings = np.load('emb/Node2vec/node_%s.npy' % filename)
			graph_embeddings = np.load('emb/Node2vec/graph_%s.npy' % filename)
		except:
			for g in G:
				target_vertex_uid = int(g.graph['target_uid'])
				target_vertex = [x for x,y in g.nodes(data=True) if y['name']==str(target_vertex_uid)]
				target_vertex = int(target_vertex[0])
				model = Node2Vec(g, **params)
				model = model.fit(window=10, min_count=1, batch_words=4)

				for vertex_id in range(len(g)):
					X = model.wv[str(vertex_id)]
					g_embedding = [0]*len(X)
					#save embedding of the targeted node
					if vertex_id == target_vertex:
						embeddings.append(list(X))

					#global graph embedding (average of all nodes)
					for f_id in range(len(X)):
						g_embedding[f_id] += float(X[f_id])
				g_emb = [g_embedding[x]/float(len(X)) for x in range(len(g_embedding))]
				graph_embeddings.append(g_emb)

	else:
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

				#global graph embedding (average of all nodes)
				for f_id in range(len(X)):
					g_embedding[f_id] += float(X[f_id])
			g_emb = [g_embedding[x]/float(len(X)) for x in range(len(g_embedding))]
			graph_embeddings.append(g_emb)

	if timer:
		print(" == Node2vec Runtime : %s s ==" % (time.time() - start_time))
	np.save('emb/Node2vec/node_%s.npy' % filename, embeddings)
	np.save('emb/Node2vec/graph_%s.npy' % filename, graph_embeddings)
	return embeddings, graph_embeddings 

'''
Generates and saves FGSD embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: List of embeddings (1 per graph in G)
'''
def gen_FGSD(G, parameters, regenerate=False, timer=True):

	try:
		params = parameters["fgsd"]
	except:
		params = {}

	#fsgd works with undirected graphs
	for i in range(len(G)):
		G[i] = G[i].to_undirected()

	start_time = time.time()
	filename = "FGSD" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			X = np.load('emb/FGSD/%s.npy' % filename)
		except:
			model = FGSD(**params)
			model.fit(G)
			X = model.get_embedding()
			np.save('emb/FGSD/%s.npy' % filename, X)
	else:
		model = FGSD(**params)
		model.fit(G)
		X = model.get_embedding()
		np.save('emb/FGSD/%s.npy' % filename, X)

	if timer:
		print(" == FGSD Runtime : %s s ==" % (time.time() - start_time))
	return X

'''
Generates and saves SF embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: List of embeddings (1 per graph in G)
'''
def gen_SF(G_train, G_test, parameters, regenerate=False, timer=True):

	G = G_train + G_test
	try:
		params = parameters["sf"]
	except:
		params = {}

	start_time = time.time()
	filename = "SF_WP" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			X = np.load('emb/SF/%s.npy' % filename)
		except:
			model = SF(**params)
			model.fit(G)
			X = model.get_embedding()
			np.save('emb/SF/%s.npy' % filename, X)
	else:
		model = SF(**params)
		model.fit(G)
		X = model.get_embedding()
		np.save('emb/SF/%s.npy' % filename, X)

	if timer:
		print(" == SF Runtime : %s s ==" % (time.time() - start_time))

	X_train = X[:len(G_train)]
	X_test = X[len(G_train):]
	return X_train, X_test

'''
Generates and saves GL2Vec embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: List of embeddings (1 per graph in G)
'''
def gen_GL2Vec(G, parameters, regenerate=False, timer=True):

	try:
		params = parameters["gl2vec"]
	except:
		params = {}

	#fsgd works with undirected graphs
	for i in range(len(G)):
		G[i] = G[i].to_undirected()

	start_time = time.time()
	filename = "GL2Vec" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			X = np.load('emb/GL2Vec/%s.npy' % filename)
		except:
			model = GL2Vec(**params)
			model.fit(G)
			X = model.get_embedding()
			np.save('emb/GL2Vec/%s.npy' % filename, X)
	else:
		model = GL2Vec(**params)
		model.fit(G)
		X = model.get_embedding()
		np.save('emb/GL2Vec/%s.npy' % filename, X)

	if timer:
		print(" == GL2Vec Runtime : %s s ==" % (time.time() - start_time))
	return X

'''
Generates and saves NetLSD embeddings in a file. If the embedding already exists in a file, loads it (except if regenerate=True).
G: List of graphs
parameters: Dictionnary containing parameters for all embedding methods. Indexs are methods names and values are dictionnaries of parameters.
regenerate: Boolean to indicates whether the embedding should be regenerated or loaded from file (if exists).
timer: Boolean to print the total runtime of the method.

Return: List of embeddings (1 per graph in G)
'''
def gen_NetLSD(G, parameters, regenerate=False, timer=True):

	try:
		params = parameters["netlsd"]
	except:
		params = {}

	#fsgd works with undirected graphs
	for i in range(len(G)):
		G[i] = G[i].to_undirected()

	start_time = time.time()
	filename = "NetLSD" + str(len(G))
	for k in params:
		filename += "_" + str(k) + str(params[k])

	if regenerate is False:
		try:
			X = np.load('emb/NetLSD/%s.npy' % filename)
		except:
			model = NetLSD(**params)
			model.fit(G)
			X = model.get_embedding()
			np.save('emb/NetLSD/%s.npy' % filename, X)
	else:
		model = NetLSD(**params)
		model.fit(G)
		X = model.get_embedding()
		np.save('emb/NetLSD/%s.npy' % filename, X)

	if timer:
		print(" == NetLSD Runtime : %s s ==" % (time.time() - start_time))
	return X


def camembert_embeddings():
	X = np.load('emb/text/camembert_emb_sentences.npy', allow_pickle=True)
	return X

def flaubert_embeddings():
	X = np.load('emb/text/flaubert_emb_sentences.npy', allow_pickle=True)
	return X

if __name__ == "__main__":

	p = argparse.ArgumentParser()
	p.add_argument("--corpus", type=str, help='corpus to use: complet or equilibre')
	p.add_argument("--type", type=str)
	p.add_argument("--timer", action="store_true")
	p.add_argument("--regenerate", action="store_true")
	p.add_argument("--graph2vec", action="store_true")
	p.add_argument("--fgsd", action="store_true")
	p.add_argument("--sf", action="store_true")
	p.add_argument("--gl2vec", action="store_true")
	p.add_argument("--netlsd", action="store_true")
	p.add_argument("--node2vec", action="store_true")
	p.add_argument("--boostne", action="store_true")
	p.add_argument("--walklets", action="store_true")
	p.add_argument("--deepwalk", action="store_true")
	p.add_argument("--nmfadmm", action="store_true")
	p.add_argument("--graphwave", action="store_true")
	p.add_argument("--grarep", action="store_true")
	p.add_argument("--nodesketch", action="store_true")
	p.add_argument("--netmf", action="store_true")
	p.add_argument("--role2vec", action="store_true")
	p.add_argument("--camembert", action="store_true")
	p.add_argument("--flaubert", action="store_true")
	p.add_argument("--combine", action="store_true")
	args = p.parse_args()

	params = parse("config.txt")
	G_train, G_test, Y_train, Y_test = load_graphs()


	features = []

	if args.graph2vec:
		X_train, X_test = gen_Graph2vec(G_train, G_test, params, regenerate=args.regenerate, timer=args.timer)
		print ("Graph2vec")
		compute_res(X_train, X_test, Y_train, Y_test)

	if args.fgsd:
		X = gen_FGSD(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X)
		else:
			X = split(X)
			print ("FGSD")
			compute_CV(X, labels)

	if args.sf:
		X_train, X_test = gen_SF(G_train, G_test, params, regenerate=args.regenerate, timer=args.timer)
		print ("SF")
		compute_res(X_train, X_test, Y_train, Y_test)

	if args.netlsd:
		X = gen_NetLSD(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X)
		else:
			X = split(X)
			print ("NetLSD")
			compute_CV(X, labels)

	if args.gl2vec:
		X = gen_GL2Vec(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X)
		else:
			X = split(X)
			print ("GL2Vec")
			compute_CV(X, labels)

	if args.boostne:
		X_boostne_node, X_boostne_graph = gen_BoostNE(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X_boostne_node)
		else:
			X_boostne_node = split(X_boostne_node)
			X_boostne_graph = split(X_boostne_graph)
			print ("BoostNE")
			compute_CV(X_boostne_node, labels)
			compute_CV(X_boostne_graph, labels)
	
	if args.walklets:
		X_walklets_node, X_walklets_graph = gen_Walklets(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X_walklets_node)
		else:
			X_walklets_node = split(X_walklets_node)
			X_walklets_graph = split(X_walklets_graph)
			print ("Walklets")
			compute_CV(X_walklets_node, labels)
			compute_CV(X_walklets_graph, labels)

	if args.deepwalk:
		X_deepwalk_node, X_deepwalk_graph = gen_DeepWalk(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X_deepwalk_node)
		else:
			X_deepwalk_node = split(X_deepwalk_node)
			X_deepwalk_graph = split(X_deepwalk_graph)
			print ("Deepwalk")
			compute_CV(X_deepwalk_node, labels)
			compute_CV(X_deepwalk_graph, labels)

	if args.nmfadmm:
		X_NMFADMM_node, X_NMFADMM_graph = gen_NMFADMM(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X_NMFADMM_node)
		else:
			X_NMFADMM_node = split(X_NMFADMM_node)
			X_NMFADMM_graph = split(X_NMFADMM_graph)
			print ("NMFADMM")
			compute_CV(X_NMFADMM_node, labels)
			compute_CV(X_NMFADMM_graph, labels)

	if args.graphwave:
		X_GraphWave_node, X_GraphWave_graph = gen_GraphWave(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X_GraphWave_node)
		else:
			X_GraphWave_node = split(X_GraphWave_node)
			X_GraphWave_graph = split(X_GraphWave_graph)
			print ("GraphWave")
			compute_CV(X_GraphWave_node, labels)
			compute_CV(X_GraphWave_graph, labels)
	
	if args.node2vec:
		X_Node2vec_node, X_Node2vec_graph = gen_Node2vec(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X_Node2vec_node)
		else:
			X_Node2vec_node = split(X_Node2vec_node)
			X_Node2vec_graph = split(X_Node2vec_graph)
			print ("Node2vec")
			compute_CV(X_Node2vec_node, labels)
			compute_CV(X_Node2vec_graph, labels)

	if args.grarep:
		X_GraRep_node, X_GraRep_graph = gen_GraRep(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X_GraRep_node)
		else:
			X_GraRep_node = split(X_GraRep_node)
			X_GraRep_graph = split(X_GraRep_graph)
			print ("GraRep")
			compute_CV(X_GraRep_node, labels)
			compute_CV(X_GraRep_graph, labels)

	if args.nodesketch:
		X_NodeSketch_node, X_NodeSketch_graph = gen_NodeSketch(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X_NodeSketch_node)
		else:
			X_NodeSketch_node = split(X_NodeSketch_node)
			X_NodeSketch_graph = split(X_NodeSketch_graph)
			print ("NodeSketch")
			compute_CV(X_NodeSketch_node, labels)
			compute_CV(X_NodeSketch_graph, labels)

	if args.netmf:
		X_NetMF_node, X_NetMF_graph = gen_NetMF(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X_NetMF_node)
		else:
			X_NetMF_node = split(X_NetMF_node)
			X_NetMF_graph = split(X_NetMF_graph)
			print ("NetMF")
			compute_CV(X_NetMF_node, labels)
			compute_CV(X_NetMF_graph, labels)

	if args.role2vec:
		X_Role2Vec_node, X_Role2Vec_graph = gen_Role2Vec(G, params, regenerate=args.regenerate, timer=args.timer)
		if args.combine:
			features.append(X_Role2Vec_node)
		else:
			X_Role2Vec_node = split(X_Role2Vec_node)
			X_Role2Vec_graph = split(X_Role2Vec_graph)
			print ("Role2Vec")
			compute_CV(X_Role2Vec_node, labels)
			compute_CV(X_Role2Vec_graph, labels)
	
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
		X = split(X)
		fusion_CV(X, labels)
	
	
