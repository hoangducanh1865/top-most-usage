import topmost 
import torch

from topmost import eva, Preprocess
from src.config.config import DEVICE


# Download a preprocessed dataset
topmost.download_dataset("20NG", cache_path="./datasets")


# Train a model
dataset = topmost.BasicDataset(".dataset/20NG", device=DEVICE, read_labels=True)

model = topmost.ProdLDA(dataset.vocab_size)
model = model.to(DEVICE)

trainer = topmost.BasicTrainer(model, dataset)

top_words, train_theta = trainer.train()   


# Evaluate
TD = eva._diversity(top_words)
TC = eva._coherence(dataset.train_texts, dataset.vocab, top_words)

test_theta = trainer.test(dataset.test_data)
clustering_results = eva._clustering(test_theta, dataset.test_labels)
classification_results = eva._cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)


# Test new documents
new_docs = [
    "This is a new document about space, including words like space, satellite, launch, orbit.",
    "This is a new document about Microsoft Windows, including words like windows, files, dos."
]

preprocess = Preprocess()
new_parsed_docs, new_bow = preprocess.parse(new_docs, vocab=dataset.vocab)
new_theta = trainer.test(torch.as_tensor(new_bow.toarray(), device=DEVICE).float())