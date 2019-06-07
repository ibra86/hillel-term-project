class ZNormalizer:
    eps = 1e-8

    def fit(self, x):
        self.mean = x.mean(0)
        self.std = x.std(0)

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)