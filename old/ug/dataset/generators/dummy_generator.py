import random


class DummyGenerator:
    def __init__(self, k, num_of_samples):
        self.k = k
        self.num_of_samples = num_of_samples

    def __call__(self, x_set, y_set, *args, **kwargs):
        cls_dict = {}
        x = []
        y = []

        for x, y in zip(x_set, y_set):
            if cls_dict.get(y) is None:
                cls_dict[y] = []
            cls_dict[y].append(x)

        classes = list(cls_dict.keys())

        for _ in range(self.k):
            sample_classes = random.sample(classes, 4)
            sample = []
            for c in sample_classes:
                sample.append(random.sample(cls_dict[c], 1)[0])
            x.append(sample)
            y.append(sample_classes)

        return x, y
