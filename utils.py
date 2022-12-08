def compute_accuracy(posteriors, labels):
    predictions = posteriors.argmax(dim=1)
    num_correct = (predictions == labels).sum()
    accuracy = num_correct / labels.size(dim=0)
    return accuracy
