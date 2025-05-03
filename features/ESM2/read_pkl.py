import pickle

with open("features/ESM2/output_embeds.pkl", "rb") as f:
    esm_dict = pickle.load(f)

print(esm_dict)