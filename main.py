import numpy as np

# Constants
DATA_PATH = 'movie_ratings'

def load_data(file_path: str) -> np.ndarray:
    with open(file_path, 'r') as file:
        data = file.readlines()
    return np.array(data)

print(load_data(DATA_PATH))
