from flair.data import Sentence
from flair.embeddings import SentenceTransformerDocumentEmbeddings

import numpy as np


embedding_cam = SentenceTransformerDocumentEmbeddings('camembert-base')
embedding_flau = SentenceTransformerDocumentEmbeddings('flaubert/flaubert_base_cased')

camembert_embeddings = []
flaubert_embeddings = []
dataset_length = 1
for i in range(dataset_length):
	message = open("text_conversations/conv_%s.txt" % i).read()
	
	sentence = Sentence(message)
	embedding_cam.embed(sentence)
	camembert_embeddings.append(sentence.embedding.numpy())
	embedding_flau.embed(sentence)
	flaubert_embeddings.append(sentence.embedding.numpy())

np.save('emb/text/camembert_emb_sentences.npy', camembert_embeddings)
np.save('emb/text/flaubert_emb_sentences.npy', flaubert_embeddings)
