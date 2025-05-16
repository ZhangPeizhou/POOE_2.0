# features/get_features.py
import pickle
import os

class Features:
    def __init__(self, info):
        # info 通常是 ["esm2", "output_by_fasta", "1670"]
        self.data_path = os.path.join("features", *info)
        self.data_dict = {}
        print("[DEBUG] Current resolved feature path:", self.data_path)

        # 加载 .pkl 文件（可能有多个）
        for filename in os.listdir(self.data_path):
            if filename.endswith(".pkl"):
                with open(os.path.join(self.data_path, filename), 'rb') as f:
                    self.data_dict.update(pickle.load(f))

    def get(self, name, foldn=None):
        if name not in self.data_dict:
            raise KeyError(f"Protein name '{name}' not found in features.")
        return self.data_dict[name]
    
    

