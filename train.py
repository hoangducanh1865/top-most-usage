from src.data.dataset_utils import get_dataset
from src.models.model import get_model_and_trainer

def train_model():
    dataset = get_dataset()
    model, trainer = get_model_and_trainer(dataset)
    top_words, train_theta = trainer.train()   
    return dataset, model, trainer, top_words, train_theta