from uvnet.models import Contrast
from retrieval.vector_db import VectorDatabase
from dgl.data.utils import load_graphs
from torch import FloatTensor

model_weights = "./results/0619/133643/best.ckpt"
vector_db_folder = "./data/vec_db"
vector_db_name = "FaBWave"
cad_file = "./data/FABWave/Socket_Head_Screws/bin/jazukassocketheadscrew10.bin" # bin file


db = VectorDatabase(vector_db_folder, vector_db_name)
model = Contrast.load_from_checkpoint(model_weights)


graph = load_graphs(cad_file)[0][0]
graph.ndata["x"] = graph.ndata["x"].type(FloatTensor)
graph.edata["x"] = graph.edata["x"].type(FloatTensor)

query_vector = model.predict_one(graph)

retrieval_topk = db.search(query_vector, k=7)

a = 0