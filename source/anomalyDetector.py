import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np


class AnomalyDetector(Model):
    
    def __init__(self):
        super(AnomalyDetector,self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32,activation="relu"),
            layers.Dense(16,activation="relu"),
            layers.Dense(8,activation="relu")])
        
        self.decoder = tf.keras.Sequential([
            layers.Dense(16,activation="relu"),
            layers.Dense(32,activation="relu"),
            layers.Dense(140,activation="sigmoid")])
        
    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
def anomalyPredict(model: AnomalyDetector, data: np.array, threshold: float):
    """
    uses the anomalyDetector model to predict outliers for a given dataset and threshold

    Parameters:
    -----------
        model : AnomalyDetector 
            autoencoder model 

        data : np.array
            data to detect anomalies in

        threshold : float
            threshold value 
    """
    
    reconstructions = model(data)
    loss = tf.keras.losses.mae(reconstructions,data)
    return tf.math.less(loss,threshold)

def print_stats(predictions,labels):
    """
    Prints relevant stats and scores for anomaly classification

    Parameters
    ----------
        predictions : array
            predictions from anomalyPredict

        labels : array
            test labels for verficiation purposes
    """
    print(f"Accuracy = {accuracy_score(labels,predictions)}")
    print(f"Precision = {precision_score(labels,predictions)}")
    print(f"Recall = {recall_score(labels,predictions)}")