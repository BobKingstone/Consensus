
"""
    TODO
    Collect training data - done
    Get handle on clients -done
    Generate collection of weights 
        evolve until best result found
    run test and then mutate weights 
"""  

import copy
from io import BytesIO

import numpy as np
import grpc
from config.client_config import ConfigHelper
from helpers.evaluators import EvaluatorEntry
from repository.sample_db import SampleRepository
from ensemblers.participant_response import ParticipantResponse

import evaluator_pb2
import evaluator_pb2_grpc
from evaluator_pb2 import EvaluationRequest


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



class Individual_Consensus():
    """
        Calculates the DE weights for the participants
    """
    def __init__(self, weights):
        self.weights = copy.deepcopy(weights)
        self.__evaluators = []
        self.__sample_repo = SampleRepository()
        self._load_evaluators()
        self.accuracy = 0
        self.total_seen_count = 0
        self.accuracy = 0


    def _load_evaluators(self):
        """Loads the local config file of evaluation servers."""
        evaluators = ConfigHelper.get_config_section_values('Servers', 'evaluators').split()
        i = 0
        for e in evaluators:
            i += 1
            self._add_evaluator(EvaluatorEntry(i, str(e)))


    def _add_evaluator(self, instance):
        """Add record of evaltaion server."""
        self.__evaluators.append(instance)


    def _get_predictions(self, sample):
        """Calculates the predicted class using the objects defined ensembler method."""
        evaluator_predictions = []
            # get evaluator predictions for this sample
        for evaluator in self.__evaluators:
            with grpc.insecure_channel(evaluator.address) as channel:
                stub = evaluator_pb2_grpc.EvaluatorStub(channel)
                # make the request to the evaluator.
                response = stub.GetEvaluation(ndarray_to_proto(sample))
                # process the response.
                evaluator_experience = response.experience
                evaluator_prediction = proto_to_ndarray(response)
                evaluator_predictions.append(ParticipantResponse(evaluator_prediction, evaluator_experience))
        
        values = np.array(self._extract_values(evaluator_predictions))
        return values
    

    def _extract_values(self, participant_response:ParticipantResponse):
        """Retrieves the numpy response array from the participant response."""
        values= []
        for x in participant_response:
            values.append(x.get_prediction())
        
        return values
    

    def _calculated_weighted_predicitions(self, predicitions):
        # multiple the preidctions by the weights
        weighted_predicitions = []
        j = 0
        print(predicitions)
        for i in predicitions:
            results.append(weighted_predicitions[j] * self.weights['agent_' + str(j + 1)])

        return results


    def _update_accuracy(self, prediction, actual):
        """ Records the accuracy of this ensembler run"""
        self.total_seen_count += 1
        if actual[0] == 'normal' and prediction[0] == 0:
            self.accuracy += 1
        elif actual[0] != 'normal' and prediction[0] == 1:
            self.accuracy += 1


    def init_current(self, r_state):
        next_state = 0
        for par in self.weights.keys():
            state = r_state + next_state
            if self.weights[par]["val_range"].dtype == np.int:
                self.weights[par]["current"] = np.random.RandomState(state).randint(
                    low=self.weights[par]["val_range"][0],
                    high=self.weights[par]["val_range"][1]) 
            else:
                self.weights[par]["current"] = np.random.RandomState(state).uniform(
                    low=self.weights[par]["val_range"][0],
                    high=self.weights[par]["val_range"][1])
            next_state += 1


    def get_accuracy(self):
        return (self.accuracy / self.total_seen_count * 100)

    def train(self, rounds : int):
        """
        Uses the given weights by params id for.
        each participant as each will be in the same order
        calculate mean of weights - weights are passed in by DE not experieince.
        apply stddev of weights to predictions then 
        argmax for columsn = result.
        """

        # loop through rounds
        i = 0
        while i < rounds:
            sample, _ = self.__sample_repo.get_next_sample()
            prediction_raw = self._get_predictions(sample)
            for x in prediction_raw:
                print("Prediction (0,0)", x)
            
            weighted_predictions = self._calculated_weighted_predicitions(prediction_raw)
            result = np.argmax(weighted_predictions, axis=1)
            self._update_accuracy(result, self.__sample_repo.get_last_sample_type())

            i += 1

        # using given weights calculate accuracy
        return (self.accuracy / self.total_seen_count * 100)
