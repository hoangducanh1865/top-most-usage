import torch 
from topmost import Preprocess
from src.config.config import DEVICE

def infer_new_docs(trainer, dataset):
    new_docs = [
        "This is a new document about space, including words like space, satellite, launch, orbit.",
        "This is a new document about Microsoft Windows, including words like windows, files, dos."
    ]
    preprocess = Preprocess()
    _, new_bow = preprocess.parse(new_docs, vocab=dataset.vocab)
    new_theta = trainer.test(torch.as_tensor(new_bow.toarray(), device=DEVICE).float())
    print("New docs topic distribution:", new_theta)
