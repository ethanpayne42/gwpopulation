"""
Implemented redshift models
"""

import numpy as np

from ..cupy_utils import to_numpy, trapz, xp
from scipy.interpolate import interp1d


class _Redshift(object):
    """
    Base class for models which include a term like dVc/dz / (1 + z)
    """

    def __init__(self, z_max=2.3):
        from astropy.cosmology import Planck15

        self.z_max = z_max
        self.zs_ = np.linspace(1e-3, z_max, 1000)
        self.zs = xp.asarray(self.zs_)
        self.dvc_dz_ = Planck15.differential_comoving_volume(self.zs_).value * 4 * np.pi
        self.dvc_dz = xp.asarray(self.dvc_dz_)
        self.cached_dvc_dz = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _cache_dvc_dz(self, redshifts):
        self.cached_dvc_dz = xp.asarray(
            np.interp(to_numpy(redshifts), self.zs_, self.dvc_dz_, left=0, right=0)
        )

    def normalisation(self, parameters):
        r"""
        Compute the normalization or differential spacetime volume.

        .. math::
            \mathcal{V} = \int dz \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)

        Parameters
        ----------
        parameters: dict
            Dictionary of parameters

        Returns
        -------
        (float, array-like): Total spacetime volume
        """
        psi_of_z = self.psi_of_z(redshift=self.zs, **parameters)
        norm = trapz(psi_of_z * self.dvc_dz / (1 + self.zs), self.zs)
        return norm

    def probability(self, dataset, **parameters):
        normalisation = self.normalisation(parameters=parameters)
        differential_volume = self.differential_spacetime_volume(
            dataset=dataset, **parameters
        )
        in_bounds = dataset["redshift"] <= self.z_max
        return differential_volume / normalisation * in_bounds

    def psi_of_z(self, redshift, **parameters):
        raise NotImplementedError

    def differential_spacetime_volume(self, dataset, **parameters):
        r"""
        Compute the differential spacetime volume.

        .. math::
            d\mathcal{V} = \frac{1}{1+z} \frac{dVc}{dz} \psi(z|\Lambda)

        Parameters
        ----------
        dataset: dict
            Dictionary containing entry "redshift"
        parameters: dict
            Dictionary of parameters
        Returns
        -------
        differential_volume: (float, array-like)
            Differential spacetime volume
        """
        psi_of_z = self.psi_of_z(redshift=dataset["redshift"], **parameters)
        differential_volume = psi_of_z / (1 + dataset["redshift"])
        try:
            differential_volume *= self.cached_dvc_dz
        except (TypeError, ValueError):
            self._cache_dvc_dz(dataset["redshift"])
            differential_volume *= self.cached_dvc_dz
        return differential_volume


class PowerLawRedshift(_Redshift):
    r"""
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270

    .. math::
        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)

        \psi(z|\gamma, \kappa, z_p) &= (1 + z)^\lambda

    Parameters
    ----------
    lamb: float
        The spectral index.
    """

    def __call__(self, dataset, lamb):
        return self.probability(dataset=dataset, lamb=lamb)

    def psi_of_z(self, redshift, **parameters):
        return (1 + redshift) ** parameters["lamb"]

class BrokenPowerLawRedshift(_Redshift):
    r"""
    Broken power law

    .. math::
        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\lambda_1, \lambda_2, z_break)

        \psi(z|\lambda_1, \lambda_2, z_break) &= (1 + z)^\lambda_1 for z < z_break
                                              &= (1 + z_break)^(\lambda_1 - \lambda_2) * (1 + z)^\lambda_2 for z_break < z < z_max

    Parameters
    ----------
    lamb1: float
        The spectral index of the low redshift component.
    lamb2: float
        The spectral index of the high redshift component.
    z_break: float
        Break location of the redshift
    """

    def __call__(self, dataset, lamb1, lamb2, z_break):
        return self.probability(dataset=dataset, lamb1=lamb1, lamb2=lamb2, z_break=z_break)

    def psi_of_z(self, redshift, **parameters):
        return (1 + redshift) ** parameters["lamb1"] * (redshift < parameters['z_break']) + (1 + parameters['z_break'])**(parameters["lamb1"] - parameters["lamb2"]) * (1 + redshift)**parameters['lamb2'] * (redshift > parameters['z_break'])

        
class MadauDickinsonRedshift(_Redshift):
    r"""
    Redshift model from Fishbach+ https://arxiv.org/abs/1805.10270 (33)
    See https://arxiv.org/abs/2003.12152 (2) for the normalisation

    The parameterisation differs a little from there, we use

    .. math::
        p(z|\gamma, \kappa, z_p) &\propto \frac{1}{1 + z}\frac{dV_c}{dz} \psi(z|\gamma, \kappa, z_p)

        \psi(z|\gamma, \kappa, z_p) &= \frac{(1 + z)^\gamma}{1 + (\frac{1 + z}{1 + z_p})^\kappa}

    Parameters
    ----------
    gamma: float
        Slope of the distribution at low redshift
    kappa: float
        Slope of the distribution at high redshift
    z_peak: float
        Redshift at which the distribution peaks.
    z_max: float, optional
        The maximum redshift allowed.
    """

    def __call__(self, dataset, gamma, kappa, z_peak):
        return self.probability(
            dataset=dataset, gamma=gamma, kappa=kappa, z_peak=z_peak
        )

    def psi_of_z(self, redshift, **parameters):
        gamma = parameters["gamma"]
        kappa = parameters["kappa"]
        z_peak = parameters["z_peak"]
        psi_of_z = (1 + redshift) ** gamma / (
            1 + ((1 + redshift) / (1 + z_peak)) ** kappa
        )
        psi_of_z *= 1 + (1 + z_peak) ** (-kappa)
        return psi_of_z


def total_four_volume(lamb, analysis_time, max_redshift=2.3):
    from astropy.cosmology import Planck15

    redshifts = np.linspace(0, max_redshift, 1000)
    psi_of_z = (1 + redshifts) ** lamb
    normalization = 4 * np.pi / 1e9 * analysis_time
    total_volume = (
        np.trapz(
            Planck15.differential_comoving_volume(redshifts).value
            / (1 + redshifts)
            * psi_of_z,
            redshifts,
        )
        * normalization
    )
    return total_volume

def psi_of_z_powerlaw(z, **kwargs):
    return (1+z)**kwargs['lamb']
    
class BaseInterpolatedPowerlaw(_Redshift):
    '''
    Base class for the Interpolated Powerlaw classes (vary the number of nodes) 
    '''
    primary_model = psi_of_z_powerlaw

    def __init__(self, nodes=10, kind='cubic', z_max=1.9):
        super(BaseInterpolatedPowerlaw, self).__init__(z_max=z_max)
        self.norm_selector = None # store selector array for normalizations since spline knots stay fixed
        self.spline_selector = None # store spline selector array since knots stay fixed
        self.spline = None # store the spline interpolant so that intpolation only happens once in p_m1 and NOT again in norm_pm1 
        self.kind = kind # can change to different types of interpolation supported with scipy.interpolate.interp1d
        self.nodes = nodes # store number of knots (nodes) which is changed within each subclass

    def psi_of_z(self, redshift, **kwargs):
        z_splines = [kwargs.pop(f'z{i}') for i in range(self.nodes)]
        f_splines = [kwargs.pop(f'f{i}') for i in range(self.nodes)]
        
        # construct selector arrays if first call (THIS WOULD NEED CHANGED IF NOT USING FIXED KNOT LOCATIONS)
        if self.spline_selector is None:
            self.spline_selector = (redshift >= z_splines[0]) & (redshift <= z_splines[-1])
        if self.norm_selector is None:
            self.norm_selector = (self.zs >= z_splines[0]) & (self.zs <= z_splines[-1])
        
        # Create the spline interpolant from knot values
        self.spline = interp1d(z_splines, f_splines, kind=self.kind)

        # Construct powerlaw
        psi_of_z_values = self.__class__.primary_model(redshift, **kwargs)
        
        # Apply perturbation
        perturbation = self.spline(redshift[self.spline_selector])
        psi_of_z_values[self.spline_selector] *= xp.exp(perturbation)
        
        return psi_of_z_values
    
    def __call__(self, dataset, lamb, z0, z1, z2, z3, z4, z5, z6, z7, z8, z9,
                 f0, f1, f2, f3, f4, f5, f6, f7, f8, f9):
        return self.probability(dataset=dataset,
                  lamb=lamb, z0=z0, z1=z1, z2=z2, z3=z3, z4=z4, z5=z5,
                  z6=z6, z7=z7, z8=z8, z9=z9, f0=f0, f1=f1, f2=f2, f3=f3, f4=f4,
                  f5=f5,f6=f6, f7=f7, f8=f8, f9=f9)
    

class InterpolatedPowerlaw10(BaseInterpolatedPowerlaw):
    '''
    Subclass of the Base Interpolated Powerlaw to use 10 knots.  __call__ method needs args explictly written for bilby.hyper.model.Model() 
    class to know which parameters go with each model
    '''
    def __init__(self, nodes=10, kind='cubic', z_max=1.9):
        return super(InterpolatedPowerlaw10, self).__init__(nodes=nodes, kind=kind, z_max=z_max)
    
    def __call__(self, dataset, lamb, z0, z1, z2, z3, z4, z5, z6, z7, z8, z9,
                 f0, f1, f2, f3, f4, f5, f6, f7, f8, f9):
        return super(InterpolatedPowerlaw10, self).__call__(dataset=dataset,
                                                          lamb=lamb, z0=z0, z1=z1, z2=z2, z3=z3, z4=z4, z5=z5,
                                                          z6=z6, z7=z7, z8=z8, z9=z9, f0=f0, f1=f1, f2=f2, f3=f3, f4=f4,
                                                          f5=f5,f6=f6, f7=f7, f8=f8, f9=f9)
