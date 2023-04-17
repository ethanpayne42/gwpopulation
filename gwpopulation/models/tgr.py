from ..cupy_utils import xp


class Gaussian(object):
    """
    Simple class for the Gaussian hierarchical model for TGR parameters
    """
    
    variable_names = ['mu_tgr', 'std_tgr']
    
    def __init__(self):
        pass
    
    
    def __call__(self, dataset, mu_tgr, std_tgr):

        norm = xp.sqrt(2*xp.pi*std_tgr**2)
        prob = xp.exp(-xp.power(dataset['dphi'] - mu_tgr, 2) / (2 * std_tgr**2)) / norm
        return prob