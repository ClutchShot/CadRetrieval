import faiss
import numpy as np
import json
import os
import hashlib
from typing import List, Dict, Any, Union, Optional
import pickle
from collections import defaultdict

class VectorDatabase:
    def __init__(self, db_folder: str, db_name: str, dimension: int = 64, metric='L2'):
        """
        Enhanced vector database with name and label support

        Args:
            db_folder: Folder path for database files
            db_name: Base name for database files
            dimension: Dimension of feature vectors
        """
        self.db_folder = db_folder
        self.db_name = db_name
        self.dimension = dimension
        self.metric = metric
        self.index = None
        self.metadata = []
        self.vector_hashes = set()

        # Create folder if not exists
        os.makedirs(self.db_folder, exist_ok=True)

        # File paths
        self.index_path = os.path.join(self.db_folder, f"{db_name}.faiss")
        self.meta_path = os.path.join(self.db_folder, f"{db_name}_meta.pkl")

        self._initialize_database()

    def _initialize_database(self):
        """Initialize or load database with security checks"""
        if os.path.exists(self.index_path):
            self._load_database()
        else:
            self._create_new_database()

    def _create_new_database(self):
        """Initialize a new FAISS index"""

        if self.metric == 'L2':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.metric == 'IP':
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError("Unsupported metric. Use 'L2' or 'IP'")

    def _load_database(self):
        """Load existing database from disk"""

        # Load index
        self.index = faiss.read_index(self.index_path)

        # Load metadata and rebuild label index
        with open(self.meta_path, 'rb') as f:
            self.metadata = pickle.load(f)

        # Load vectors hashes
        for vector_index in range(self.index.ntotal):
            vector = self.index.reconstruct(vector_index)
            self.vector_hashes.add(self._get_vector_hash(vector))

    def _get_vector_hash(self, vector):
        """Create a unique hash for each vector"""
        return hashlib.sha256(vector.tobytes()).hexdigest()

    def add_vectors(self,
                    vectors: np.ndarray,
                    names: List[str],
                    labels: List[str], 
                    duplicates: bool = False):
        """
        Add vectors with names and labels to database

        Args:
            vectors: Array of feature vectors (n_vectors x dimension)
            names: List of names for each vector
            labels: List of integer labels for each vector
        """
        # Input validation
        if len(vectors) != len(names) or len(vectors) != len(labels):
            raise ValueError(
                "Vectors, names, and labels must have same length")

        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vectors must have dimension {self.dimension}")

        # Convert to numpy array if needed
        vectors = np.array(vectors, dtype='float32')

        new_metadata = []

        # Check vectors for duplicates in index and add
        for index in range(len(vectors)):

            vec_hash = self._get_vector_hash(vectors[index])
            if not duplicates and vec_hash in self.vector_hashes:
                continue
            else:
                # Add to index
                self.index.add(np.array([vectors[index]], dtype='float32'))
                # Add metadata
                new_metadata.append(
                    {
                        'name': names[index],
                        'label': labels[index],
                    }
                )
                # Add to runtim hash
                if not duplicates:
                    self.vector_hashes.add(vec_hash)

        # Add metadata
        self.metadata.extend(new_metadata)
        # Save updates
        self.save()

    def search(self,
               query: np.ndarray,
               k: int = 10,
               label_filter: Optional[Union[int, List[int]]] = None) -> List[Dict[str, Any]]:
        """
        Search database with optional label filtering

        Args:
            query: Query vector (1D array of dimension)
            k: Number of results to return
            label_filter: Optional label or list of labels to filter by

        Returns:
            List of result dictionaries with metadata and distance
        """
        if not isinstance(query, np.ndarray):
            query = np.array(query, dtype='float32')

        # Handle label filtering
        if label_filter is not None:
            if isinstance(label_filter, int):
                label_filter = [label_filter]

            # Get all vector indices for these labels
            filtered_indices = []
            for label in label_filter:
                filtered_indices.extend(self.label_index.get(label, []))

            if not filtered_indices:
                return []

            # Convert to numpy array
            filtered_indices = np.array(filtered_indices, dtype='int64')

            # Search within subset
            distances, indices = self.index.search(query, k)

            # Map back to original indices and filter
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx in filtered_indices:
                    results.append({
                        **self.metadata[idx],
                        'distance': float(dist)
                    })
                    if len(results) >= k:
                        break
            return results
        else:
            distances, indices = self.index.search(query, k)
            result = []
            for idx, dist in zip(indices, distances):
                result.append(
                    [
                        {**self.metadata[idx], 'distance': float(dist)}
                        for idx, dist in zip(idx, dist)
                        if idx >= 0
                    ]
                )

            return result


    def save(self):
        """Save database"""
        # Ensure folder exists
        os.makedirs(self.db_folder, exist_ok=True)

        # Save index
        faiss.write_index(self.index, self.index_path)

        # Save metadata
        with open(self.meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def get_vector_count(self) -> int:
        """Return number of vectors in the database"""
        return self.index.ntotal if self.index else 0
    
    def get_metadata(self) -> list:
        return self.metadata
    
    def get_metadata_dict(self):
        metadata = self.get_metadata()
        metadata_dict = dict()
        for meta in metadata:
            if meta['label'] in metadata_dict.keys():
                metadata_dict[meta['label']] += 1
            else:
                metadata_dict[meta['label']] = 1

        sorted_dict = dict(sorted(metadata_dict.items(), key=lambda item: item[1], reverse=True))
        return sorted_dict


# # Example usage
# if __name__ == "__main__":
#     # Initialize database (creates folder if needed)
#     db = VectorDatabase("data/vectcad_models", "parts_db", 128)

#     # Create sample data
#     vectors = np.random.rand(10, 128).astype('float32')
#     names = [f"part_{i}" for i in range(10)]
#     labels = [1, 2, 1, 3, 2, 1, 3, 3, 2, 1]  # Example labels

#     # Add vectors with names and labels
#     db.add_vectors(vectors, names, labels)
#     size_one = db.get_vector_count()

#     # Search without label filter
#     query = np.random.rand(2, 128).astype('float32')
#     print("All results:")
#     result = db.search(query, k=3)

#     # Search with label filter
#     print("\nOnly label 1 results:")
#     print(db.search(query, k=3, label_filter=1))

#     print("\nOnly label 1 results:")
#     print(db.search(query, k=3, label_filter=[1, 2]))

