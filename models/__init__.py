from .model import Predictor_Corrector as M
import logging 

logger = logging.getLogger('base')
def create_model(opt):
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m