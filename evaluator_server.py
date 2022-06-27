"""Evaluation server module."""
from io import BytesIO
from concurrent import futures
import logging
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random

import numpy as np
import tensorflow as tf

import grpc
from config.client_config import ConfigHelper
import evaluator_pb2
import evaluator_pb2_grpc

from evaluator_pb2 import EvaluationDecision


class EvaluationAgent():
    """NN agent wrapper"""
    def __init__(self):
        """"""
        tf.get_logger().setLevel('ERROR')
        self.__model = None
        self.load_model()


    def load_model(self):
        """Loads the given model for generating the predictions."""
        model_type = ConfigHelper.get_config_section_values('Models', 'model_type')

        if model_type is None:
            return

        model_path = ConfigHelper.get_config_section_values('Models', model_type + '_model')
        model_path = model_path + str(server_id)
        print('------------- loading model: ', model_path)
        self.__model = tf.keras.models.load_model(model_path)
        # print(self.__model.summary())


    def get_prediction(self, sample):
        """Retrieves a prediction for the given sample."""
        prediction = self.__model.predict(sample)
        return prediction


class Evaluator(evaluator_pb2_grpc.EvaluatorServicer):
    """Evaluation object for carrying out request decision making."""
    def __init__(self):
        # load the model, using the file name in the config file.
        self.__agent = EvaluationAgent()
    

    def GetEvaluation(self, request, context):
        """Returns a decision based on the message array given.
        Args:
            request (EvaluationRequest): The protobuf message to extract the sample from.
        Returns:
            Returns a EvaluationDecision.
        """
        nda = Evaluator.proto_to_ndarray(request)
        prediction_array = self.__agent.get_prediction(nda)
        calculated_prediction = np.argmax(prediction_array)
        rand_experience = random.randrange(1,100)
        # placeholder code below.
        decision = EvaluationDecision.Decision.BENIGN
        if calculated_prediction == 1:
            decision = EvaluationDecision.Decision.ANOMALY
        
        print('--------------- Published Prediction: ', decision, " --- Experience ", rand_experience)
        return Evaluator.ndarray_to_proto(decision, prediction_array, rand_experience)


    @staticmethod
    def proto_to_ndarray( nda_proto: evaluator_pb2.EvaluationRequest) -> np.ndarray:
        """Deserializes an NDArray protobuf message into a numpy array.
        Args:
            nda_proto (EvaluationRequest): protobuf message to deserialize.
        Returns:
            Returns a numpy.ndarray.
        """
        nda_bytes = BytesIO(nda_proto.ndarray)

        return np.load(nda_bytes, allow_pickle=False)
    
    @staticmethod
    def ndarray_to_proto(dec : EvaluationDecision.Decision, nda : np.ndarray, experience : int) -> EvaluationDecision:
        """Serializes a numpy array into an NDArray protobuf message.
        Args:
            nda (np.ndarray): numpy array to serialize.
        Returns:
            Returns an NDArray protobuf message.
        """
        nda_bytes = BytesIO()
        np.save(nda_bytes, nda, allow_pickle=False)

        return EvaluationDecision(decision=dec, ndarray=nda_bytes.getvalue(), experience=experience)


def serve():
    """Helper method to start the grpc server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    evaluator_pb2_grpc.add_EvaluatorServicer_to_server(Evaluator(), server)
    port = ConfigHelper.get_config_section_values('ServerPorts', 'port_' + str(server_id))
    print("Server ", server_id, " port ", port)
    server.add_insecure_port('[::]:'+ port)
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    server_id = int(sys.argv[1])
    serve()
