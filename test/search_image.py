import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

class Clip():
    def __init__(self, model_name="Bingsu/clip-vit-large-patch14-ko"):
        self.model = AutoModel.from_pretrained(model_name).to("cuda")
        self.processor = AutoProcessor.from_pretrained(model_name)

    def get_image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to("cuda")
        with torch.inference_mode():
            outputs = self.model.get_image_features(**inputs)
        return outputs[0].cpu().numpy()

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        for k in inputs:
            if isinstance(inputs[k], torch.Tensor):
                inputs[k] = inputs[k].to("cuda")
        with torch.inference_mode():
            outputs = self.model.get_text_features(**inputs)
        return outputs[0].cpu().numpy()

class SementicSearch:
    def __init__(self):
        self.clip = Clip()
        self.dump = []

    def get_text_embedding(self, text):
        return self.clip.get_text_embedding(text)

    def get_image_embedding(self, image_path):
        return self.clip.get_image_embedding(image_path)

    def add_image(self, image_path):
        emb = self.get_image_embedding(image_path)
        self.dump.append((image_path, emb))
    
    def search(self, text, topk=5):
        text_emb = self.get_text_embedding(text)
        sims = []
        for path, img_emb in self.dump:
            sim = np.dot(img_emb, text_emb) / (np.linalg.norm(img_emb) * np.linalg.norm(text_emb) + 1e-8)
            sims.append((path, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:topk]

if __name__ == "__main__":
    image_dir = "./val2017"
    searcher = SementicSearch()

    print("Indexing images...")
    for img_file in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, img_file)
        searcher.add_image(img_path)

    while True:
        query = input("Enter search query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        print(f"Searching for: {query}")
        results = searcher.search(query, topk=3)

        print("Top results:")
        for path, score in results:
            print(f"Image: {path}, Similarity: {score:.4f}")