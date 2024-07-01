# AlertEmbeddings
Abuse detection in online conversations with text and graph embeddings

## Datasets

Datasets are available at https://zenodo.org/records/11617245


## Running tests

### Signed Graph2vec
```
python src/signed_graph2vec.py --input-path "dataset/*.graphml" --output-path output/embeddings.csv
```


### SGCN
Based on https://github.com/benedekrozemberczki/SGCN

```
python SGCN/main.py --layers 96 32 --learning-rate 0.01 --reduction-dimensions 64 --epochs 10 --reduction-iterations 10 --edge-path dataset/edges.csv --embedding-path output/embedding.csv --regression-weights-path output/weights.csv
```
