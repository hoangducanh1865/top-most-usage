from train import train_model
from src.evaluate import evaluate_model
from src.infer import infer_new_docs

if __name__ == "__main__":
    dataset, model, trainer, top_words, train_theta = train_model()
    TD, TC, clustering_results, classification_results, test_theta = evaluate_model(dataset, trainer, top_words, train_theta)
    infer_new_docs(trainer, dataset)
    print("Topic Diversity:", TD)
    print("Topic Coherence:", TC)
    print("Clustering Results:", clustering_results)
    print("Classification Results:", classification_results)