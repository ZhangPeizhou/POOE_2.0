import os
import pickle

class Features:
    def __init__(self, info=None):
        # 强制使用绝对路径
        self.data_path = os.path.abspath(os.path.join("..", "features", *info))
        print(f"[DEBUG] using absolute path: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[FATAL] Path does not exist: {self.data_path}")

        self.data_dict = {}
        for filename in os.listdir(self.data_path):
            if filename.endswith(".pkl"):
                with open(os.path.join(self.data_path, filename), 'rb') as f:
                    self.data_dict.update(pickle.load(f))

    def get(self, name, foldn=None):
        if name not in self.data_dict:
            raise KeyError(f"Protein name '{name}' not found in features.")
        return self.data_dict[name]
