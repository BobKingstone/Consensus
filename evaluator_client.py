"""TODO::
1. read csv file for test samples - done
2. send first entry to evaluator - done
3. respond with actual prediction for the given sample using stored model
4. change response to include the number of samples evaluator has seen
5. change response to include reward score for the given agent
6. create different consensus implementations
7. plug in to flower and listen for training completion
8. alter consensus to allow for dropouts/timeouts and achieve consensus.
9. implement multi critic approach to allow for interconnections during training if possible.
10. Test differing consensus approaches and record values
11. write it all up!
"""

from __future__ import print_function
from __future__ import annotations

import logging
import numpy as np
import sys
from DE.DE_Collaboration import DE_Collaboration

from engine.consensus_engine import ConsensusEngine

parameters = {
    "agent_1" : {"val_range" : np.array([1, 100]).astype(float), "current" : None},
    "agent_2" : {"val_range" : np.array([1, 100]).astype(float), "current" : None},
    "agent_3" : {"val_range" : np.array([1, 100]).astype(float), "current" : None},
    "agent_4" : {"val_range" : np.array([1, 100]).astype(float), "current" : None},
    "agent_5" : {"val_range" : np.array([1, 100]).astype(float), "current" : None},
}


if __name__ == '__main__':
    logging.basicConfig()
    process_id = int(sys.argv[1])
    if process_id == 1:
        engine = ConsensusEngine()
        engine.print_evaluators()
        engine.get_ensemble_prediction(500)
        print(engine.get_accuracy())
    elif process_id == 2:
        eve = DE_Collaboration()
        eve.add_params(parameters)
        eve.evolve()
