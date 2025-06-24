from uvnet.models import Contrast
from retrieval.vector_db import VectorDatabase
from dgl.data.utils import load_graphs
from torch import FloatTensor
from datasets import util

model_weights = "./results/0619/133643/best.ckpt"
vector_db_folder = "./data/vec_db"
vector_db_name = "FaBWave"
cad_file = "./data/FABWave/Socket_Head_Screws/bin/jazukassocketheadscrew10.bin" # bin file


db = VectorDatabase(vector_db_folder, vector_db_name)
model = Contrast.load_from_checkpoint(model_weights)


graph = load_graphs(cad_file)[0][0]
# center_and_scale
graph.ndata["x"], center, scale = util.center_and_scale_uvgrid(
    graph.ndata["x"], return_center_scale=True
)
graph.edata["x"][..., :3] -= center
graph.edata["x"][..., :3] *= scale

graph.ndata["x"] = graph.ndata["x"].type(FloatTensor)
graph.edata["x"] = graph.edata["x"].type(FloatTensor)

query_vector = model.predict_one(graph).cpu().numpy()
retrieval_topk_np = db.search(query_vector, k=10)