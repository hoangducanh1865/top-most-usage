import topmost
from src.config.config import DEVICE


def get_model_and_trainer(dataset):
    model = topmost.ProdLDA(dataset.vocab_size)
    model = model.to(DEVICE)
    trainer = topmost.BasicTrainer(model, dataset)
    return model, trainer