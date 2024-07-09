AlertEmbeddings
*Abuse detection in online conversations with text and graph embeddings*

# Description
This set of scripts aims at learning various text ang graph embeddings from online conversations to detect online abuse.

# Data

The datasets are available at https://zenodo.org/records/11617245.
To use the SpaceOrigin dataset, unzip the `SpaceOrigin_graphs.zip` archive into the `in/graphs` folder.

conversation should be added in the `in/text_conversations` folder as a separate file for each conversation with each line corresponding to a message. An example is available.

# Organization
Here are the folders composing the project:
* Folder `in`: input data, including the textual conversations and graphs.
* Folder `SGCN`: set of scripts to learn embeddings using the SGCN method.
* Folder `signed_graph2vec`: set of scripts to learn embeddings using the SG2V method. 
* `text_embeddings.py`: script to learn the text embeddings.
* `unsigned_graph_embeddings.py`: script to apply te standard unsigned graph embedding models.


# Installation
This library requires python 3.8+
Dependencies car be installed with ``` pip install -r requirements.txt ```
