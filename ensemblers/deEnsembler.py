import numpy as np
from ensemblers.participant_response import ParticipantResponse


class DeEnsembler():
    """
        Ensembler using differential evolution to calculate the weights.
    """
    def get_ensemble_result(self, participant_responses):
        """returns the index of the predictions."""
        values = np.array(self.extract_values(participant_responses))
        weights = self.calculate_weights(participant_responses)

        summed_result = np.tensordot(values, weights, axes=((0),(0)))
        result = np.argmax(summed_result, axis=1)

        return result

    def extract_values(self, participant_response:ParticipantResponse):
        """Retrieves the numpy response array from the participant response."""
        values= []
        for x in participant_response:
            values.append(x.get_prediction())

        return values

    def calculate_weights(self, partiticpant_response: ParticipantResponse):
        """"""
        raw_values = []

        for x in partiticpant_response:
            raw_values.append(x.get_experience())

        return (raw_values - np.min(raw_values))/np.ptp(raw_values)