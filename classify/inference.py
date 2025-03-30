import os
# Force CPU usage and suppress GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from sentence_transformers import SentenceTransformer

class QueryClassifier:
    def __init__(self, model_path='query_classifier_model.keras'):
        # Configure TensorFlow to use CPU
        with tf.device('/CPU:0'):
            # Recreate and load model
            self.ann_model = self._create_model()
            self.ann_model.load_weights(model_path)
        
        # Initialize embedding model with CPU
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.categories = {0: 'code', 1: 'reason', 2: 'language'}
    
    def _create_model(self):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(384,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
    
    def predict(self, text):
        try:
            # Generate embedding and convert to numpy array
            embedding = self.embedding_model.encode([text], 
                                                   convert_to_tensor=False,  # Return numpy array
                                                   device='cpu')
            # Add batch dimension if needed
            if len(embedding.shape) == 1:
                embedding = np.expand_dims(embedding, axis=0)
            
            # Make prediction on CPU
            with tf.device('/CPU:0'):
                prediction = self.ann_model.predict(embedding, verbose=0)
            
            return self.categories[np.argmax(prediction)]
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return "unknown"

if __name__ == "__main__":
    # Suppress TensorFlow logging
    tf.get_logger().setLevel('ERROR')
    
    try:
        classifier = QueryClassifier()
        query = "How to implement a neural network?"
        result = classifier.predict(query)
        print(f"Query: {query}\nPredicted category: {result}")
    except Exception as e:
        print(f"Initialization error: {str(e)}")