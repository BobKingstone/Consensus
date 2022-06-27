"""Consensus engine module."""
from io import BytesIO
import numpy as np

import grpc
from ensemblers.average_ensembler import AverageEnsembler
from ensemblers.participant_response import ParticipantResponse
from ensemblers.weighted_ensembler import WeightedEnsembler
import evaluator_pb2
import evaluator_pb2_grpc
from evaluator_pb2 import EvaluationRequest

from config.client_config import ConfigHelper
from helpers.evaluators import EvaluatorEntry
from repository.sample_db import SampleRepository


# ref for these two functions comes from https://github.com/josteinbf/numproto/blob/master/numproto/numproto.py

def ndarray_to_proto(nda: np.ndarray) -> EvaluationRequest:
    """Serializes a numpy array into an NDArray protobuf message.
    Args:
        nda (np.ndarray): numpy array to serialize.
    Returns:
        Returns an NDArray protobuf message.
    """
    nda_bytes = BytesIO()
    np.save(nda_bytes, nda, allow_pickle=False)

    return EvaluationRequest(sample='note', ndarray=nda_bytes.getvalue())


def proto_to_ndarray(nda_proto: EvaluationRequest) -> np.ndarray:
    """Deserializes an NDArray protobuf message into a numpy array.
    Args:
        nda_proto (NDArray): NDArray protobuf message to deserialize.
    Returns:
        Returns a numpy.ndarray.
    """
    nda_bytes = BytesIO(nda_proto.ndarray)

    return np.load(nda_bytes, allow_pickle=False)


class ConsensusEngine():
    """Engine object that drives the collection and analysis of evaluation requests."""
    def __init__(self):
        self.ensembler = AverageEnsembler() # WeightedEnsembler()
        self.__evaluators = []
        self.__sample_repo = SampleRepository()
        self.load_evaluators()
        self.accuracy = 0
        self.total_seen_count = 0


    def add_evaluator(self, instance):
        """Add record of evaltaion server."""
        self.__evaluators.append(instance)


    def load_evaluators(self):
        """Loads the local config file of evaluation servers."""
        evaluators = ConfigHelper.get_config_section_values('Servers', 'evaluators').split()
        i = 0
        for e in evaluators:
            i += 1
            self.add_evaluator(EvaluatorEntry(i, str(e)))


    def get_ensemble_prediction(self, count : int):
        """Calculates the predicted class using the objects defined ensembler method."""

        i = 0

        while i < count:
            sample, _ = self.__sample_repo.get_next_sample()
            evaluator_predictions = []
            
            for ev in self.__evaluators:
                with grpc.insecure_channel(ev.address) as channel:
                    stub = evaluator_pb2_grpc.EvaluatorStub(channel)
                    # make the request to the evaluator.
                    response = stub.GetEvaluation(ndarray_to_proto(sample))
                    # process the response.
                    evaluator_experience = response.experience
                    evaluator_prediction = proto_to_ndarray(response)
                    evaluator_predictions.append(ParticipantResponse(evaluator_prediction, evaluator_experience))
            i += 1

            prediction = self.ensembler.get_ensemble_result(participant_responses=evaluator_predictions)
            self.update_accuracy(prediction, self.__sample_repo.get_last_sample_type())
            print(prediction, " last sample type ", self.__sample_repo.get_last_sample_type())


    def get_consensus(self, round_count):
        """Get consensus from all evaluators."""
        i = 0
        while i < round_count:
            benign = 0
            anomaly = 0

            sample, _ = self.__sample_repo.get_next_sample()

            for ev in self.__evaluators:
                with grpc.insecure_channel(ev.address) as channel:
                    stub = evaluator_pb2_grpc.EvaluatorStub(channel)
                    response = stub.GetEvaluation(ndarray_to_proto(sample))
                    if response.decision == evaluator_pb2.EvaluationDecision.Decision.BENIGN:
                        # print("Sample marked as BENIGN")
                        benign += 1
                    elif response.decision == evaluator_pb2.EvaluationDecision.Decision.ANOMALY:
                        # print("Sample marked as ANOMALY")
                        anomaly += 1
            
            print("Evaluation Results :", benign, " ", anomaly)
            print("Final Result is :", self.calculate_result(anomaly, benign))
            print("Sample was :", self.__sample_repo.get_last_sample_type())
            i += 1


    def calculate_result(self, a, b):
        """Calculate the consensus result using the given values."""
        if b > a:
            return "BENIGN"

        return "ANOMALY"


    def print_evaluators(self):
        """Helper method to print the available evaluation servers"""
        for ev in self.__evaluators:
            print(ev.id, ev.address)


    def update_accuracy(self, prediction, actual):
        """ Records the accuracy of this ensembler run"""
        self.total_seen_count += 1
        if actual[0] == 'normal' and prediction[0] == 0:
            self.accuracy += 1
        elif actual[0] != 'normal' and prediction[0] == 1:
            self.accuracy += 1


    def get_accuracy(self):
        p = self.accuracy / self.total_seen_count
        return p
