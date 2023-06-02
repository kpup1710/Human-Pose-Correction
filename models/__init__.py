from .model import Predictor_Corrector as M
import logging 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
def create_model(adj, opt):
    m = M(adj, opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m