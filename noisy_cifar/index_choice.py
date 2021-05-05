import numpy


def index_choice(labels, n, random_state):
    labels = numpy.array(labels)
    classes = numpy.unique(labels)
    n //= len(classes)

    indices = []
    for c in classes:
        class_indices = numpy.where(labels == c)[0]
        chosen_indices = random_state.choice(class_indices, n, replace=False)
        indices.extend(chosen_indices)
    return indices
