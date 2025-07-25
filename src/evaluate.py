from topmost import eva


def evaluate_model(dataset, trainer, top_words, train_theta):
    TD = eva._diversity(top_words)
    TC = eva._coherence(dataset.train_texts, dataset.vocab, top_words)
    test_theta = trainer.test(dataset.test_data)
    clustering_results = eva._clustering(test_theta, dataset.test_labels)
    classification_results = eva._cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
    return TD, TC, clustering_results, classification_results