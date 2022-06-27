"""Weighted Ensembler for model predictions."""
import numpy


from ensemblers.participant_response import ParticipantResponse


class WeightedEnsembler():
    """
    Class calculates the averageof the given model predictions returning 
    the class index of the highest value.
    """
    def get_ensemble_result(self, participant_responses):
        """returns the index of the predictions."""
        values = numpy.array(self.extract_values(participant_responses))
        weights = self.calculate_weights(participant_responses)

        # revisit this weight function as I don't think it does what it actuallly should
        summed_result = numpy.tensordot(values, weights, axes=((0),(0)))
        result = numpy.argmax(summed_result, axis=1)

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

        return (raw_values - numpy.min(raw_values))/numpy.ptp(raw_values)
