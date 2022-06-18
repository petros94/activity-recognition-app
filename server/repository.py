class InMemoryStorage:
    def __init__(self, max_len):
        self.max_len = max_len
        self.data = []

    def save_item(self, obj):
        if len(self.data) >= self.max_len:
            self.data.pop(0)
        self.data.append(obj)

    def load_item(self, idx=None):
        if idx is None:
            return self.data[-1]
        else:
            return self.data[idx]
