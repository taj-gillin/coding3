import numpy as np
import random
from tqdm import tqdm  

class MatrixFactorization:
    def __init__(self, n_users, n_movies, n_factors=20, learning_rate=0.005, 
                 regularization=0.02, n_epochs=100, verbose=True):
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.verbose = verbose
        
        self.user_factors = np.random.normal(0, 0.1, (n_users + 1, n_factors))
        self.movie_factors = np.random.normal(0, 0.1, (n_movies + 1, n_factors))
        
        self.global_bias = 0.0
        self.user_bias = np.zeros(n_users + 1)
        self.movie_bias = np.zeros(n_movies + 1)
        
    def predict(self, user_id, movie_id):
        prediction = (
            self.global_bias + 
            self.user_bias[user_id] + 
            self.movie_bias[movie_id] + 
            np.dot(self.user_factors[user_id], self.movie_factors[movie_id])
        )
        return max(0.5, min(5.0, prediction))
    
    def fit(self, ratings):
        self.global_bias = np.mean([r[2] for r in ratings])
        
        for epoch in range(self.n_epochs):
            random.shuffle(ratings)
            
            squared_error = 0
            
            for user_id, movie_id, rating in tqdm(ratings) if self.verbose else ratings:
                # Forward pass
                pred = self.predict(user_id, movie_id)

                # Backward pass
                error = rating - pred
                squared_error += error ** 2
                
                # Update biases
                self.user_bias[user_id] += self.learning_rate * (error - self.regularization * self.user_bias[user_id])
                self.movie_bias[movie_id] += self.learning_rate * (error - self.regularization * self.movie_bias[movie_id])
                
                # Update latent factors
                user_factors_update = error * self.movie_factors[movie_id] - self.regularization * self.user_factors[user_id]
                movie_factors_update = error * self.user_factors[user_id] - self.regularization * self.movie_factors[movie_id]
                self.user_factors[user_id] += self.learning_rate * user_factors_update
                self.movie_factors[movie_id] += self.learning_rate * movie_factors_update
            
            # Calculate epoch error
            rmse = np.sqrt(squared_error / len(ratings))
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs} - Error: {rmse:.4f}")
                
        return self



def read_input_file(filename):
    with open(filename, 'r') as f:
        # n, m, k
        n_users, n_movies, n_ratings = map(int, f.readline().strip().split())
        
        # Known ratings
        train_ratings = []
        for _ in range(n_ratings):
            user_id, movie_id, rating = map(float, f.readline().strip().split())
            train_ratings.append((int(user_id), int(movie_id), rating))
        
        # Number of queries
        n_queries = int(f.readline().strip())
        
        # Queries
        test_queries = []
        for _ in range(n_queries):
            user_id, movie_id = map(int, f.readline().strip().split())
            test_queries.append((user_id, movie_id))
            
    return n_users, n_movies, train_ratings, test_queries


def write_output_file(filename, predictions):
    """Write predictions to the output file"""
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

def main():
    input_file = "movie_ratings"
    output_file = "matcomp_ans"
    
    # Read input data
    print("Reading input file...")
    n_users, n_movies, train_ratings, test_queries = read_input_file(input_file)
    print(f"Data loaded: {n_users} users, {n_movies} movies, {len(train_ratings)} ratings, {len(test_queries)} queries")
     
    # Hyperparameters
    params = {
        'n_factors': 50,
        'learning_rate': 0.005,
        'regularization': 0.02,
        'n_epochs': 100
    }
    # Train final model with best parameters
    print("\nTraining model...")
    model = MatrixFactorization(n_users, n_movies, **params)
    model.fit(train_ratings)
    
    # Make predictions on test queries
    print("Making predictions on test queries...")
    predictions = [model.predict(user_id, movie_id) for user_id, movie_id in test_queries]
    
    # Write predictions to output file
    print(f"Writing predictions to {output_file}...")
    write_output_file(output_file, predictions)


if __name__ == "__main__":
    main()