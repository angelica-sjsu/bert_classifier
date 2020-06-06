from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import pickle


class EmbeddedSentences(Dataset):
    """
    Load and read embeddings for each reviews
    """
    def __init__(self, embeddings_path, labels_csv, phase):
        self.embeddings_path = embeddings_path
        self.files = os.listdir(self.embeddings_path)
        self.labels_csv = pd.read_csv(labels_csv)
        self.phase = phase

        pickle_file = f'{phase}_embeddings.pkl'
        self.data = [] 
        # pre-load embeddings into memory, speeds up training considerably
        if os.path.exists(pickle_file):
            self.data = pickle.load(open(pickle_file, 'rb'))
        else:
            for f in tqdm(self.files):
                path = os.path.join(self.embeddings_path, f)
                self.data.append(np.load(path).squeeze())
            pickle.dump(self.data, open(pickle_file, 'wb'))

    def __getitem__(self, index):
        """
        returns review embeddings and labels
        """
        label = self.labels_csv['label'][index]
        embedded = self.data[index]

        return embedded, label.astype(np.float32)

    def __len__(self):
        return len(self.files)
