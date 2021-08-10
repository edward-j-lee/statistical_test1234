import scipy.stats as stats
from scipy.stats._multivariate import multi_rv_generic, doccer, multi_rv_frozen
import numpy as np
from scipy.special import gammaln, psi, multigammaln, xlogy, entr, betaln
#https://github.com/parulsethi/scipy/commit/fe528a40cfc6014a0f4096d83852e74b637d51db?branch=fe528a40cfc6014a0f4096d83852e74b637d51db&diff=unified


_LOG_2PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)
_LOG_PI = np.log(np.pi)


class normal_invgamma_gen(multi_rv_generic):

    def __init__(self, seed=None):
        super(normal_invgamma_gen, self).__init__(seed)
        #self.__doc__ = doccer.docformat(self.__doc__, nig_docdict_params)

    def __call__(self, loc=0, variance_scale=1, shape=1, scale=1, seed=None):
        """
        Create a frozen normal inverse gamma distribution.
        See `normal_invgamma_frozen` for more information.
        """
        return normal_invgamma_frozen(loc, variance_scale, shape, scale,
                                    seed=seed)

    def _check_parameters(self, loc, variance_scale, shape, scale):
        if not np.isscalar(loc):
            raise ValueError("""loc must be a scalar""")
        if not np.isscalar(variance_scale) or variance_scale < 0:
            raise ValueError("""variance_scale must be a positive scalar""")
        if not np.isscalar(shape) or shape < 0:
            raise ValueError("""shape must be a positive scalar""")
        if not np.isscalar(scale) or scale < 0:
            raise ValueError("""scale must be a positive scalar""")
        return loc, variance_scale, shape, scale

    def _check_input(self, x, sig2):
        if x.ndim != 1:
            raise ValueError("""array must be one dimensional""")
        if sig2.ndim != 1:
            raise ValueError("""array must be one dimensional""")
        if any(sig2 < 0):
            raise ValueError("""array must consist of positive values,
                                as it represents variance""")
        return x, sig2

    def logpdf(self, x, sig2, loc=0, variance_scale=1, shape=1, scale=1):
        """
        Log of the Normal inverse gamma probability density function.
        Parameters
        ----------
        x : array
            One-dimensional array.
        sig2 : array
            One-dimensional array.
        %(_nig_doc_default_callparams)s
        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `(x, sig2)`.
        """
        loc, variance_scale, shape, scale = self._check_parameters(
                                                    loc, variance_scale, shape, scale)
        x, sig2 = self._check_input(x, sig2)

        Zinv = shape * np.log(scale) - gammaln(shape) - 0.5 * (np.log(variance_scale) + _LOG_2PI)
        out = Zinv - 0.5 * np.log(sig2) - (shape + 1.) * np.log(sig2) -\
                scale/sig2 - 0.5/(sig2 * variance_scale) * (x-loc)**2
        return out

    def pdf(self, x, sig2, loc=0, variance_scale=1, shape=1, scale=1):
        """
        Normal inverse gamma probability density function.
        Parameters
        ----------
        x : array
            One-dimensional array.
        sig2 : array
            One-dimensional array.
        %(_nig_doc_default_callparams)s
        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `(x, sig2)`.
        Notes
        -----
        %(_nig_doc_callparams_note)s
        """
        out = np.exp(self.logpdf(x, sig2, loc, variance_scale, shape, scale))
        return out

    def rvs(self, loc=0, variance_scale=1, shape=1, scale=1, size=1, random_state=None):
        """
        Draw random samples from normal and inverse gamma distributions.
        Parameters
        ----------
        %(_nig_doc_default_callparams)s
        size : integer, optional
            Number of samples to draw (default 1).
        %(_doc_random_state)s
        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `2`).
        Notes
        -----
        %(_nig_doc_callparams_note)s
        """
        loc, variance_scale, shape, scale = self._check_parameters(
                                                loc, variance_scale, shape, scale)
        random_state = self._get_random_state(random_state)

        sig2_rv = 1/random_state.gamma(shape, scale, size)
        x_rv = random_state.normal(loc, np.sqrt(sig2_rv * variance_scale), size)
        return np.array(list(zip(x_rv, sig2_rv)))

    def mean(self, loc=0, variance_scale=1, shape=1, scale=1):
        """
        Compute the mean for input random variates.
        Parameters
        ----------
        %(_nig_doc_default_callparams)s
        Returns
        -------
        (x, s) : tuple of scalar values
            Mean of the random variates
        Notes
        -----
        %(_nig_doc_callparams_note)s
        """
        loc, variance_scale, shape, scale = self._check_parameters(
                                                loc, variance_scale, shape, scale)
        x_mean = loc
        if shape > 1:
            sig2_mean = scale / (shape - 1)
        else:
            sig2_mean = np.inf
        return x_mean, sig2_mean

    def mode(self, loc=0, variance_scale=1, shape=1, scale=1):
        """
        Compute the mode for input random variates.
        Parameters
        ----------
        %(_nig_doc_default_callparams)s
        Returns
        -------
        (x, s) : tuple of scalar values
            Mode of the random variates
        Notes
        -----
        %(_nig_doc_callparams_note)s
        """
        loc, variance_scale, shape, scale = self._check_parameters(
                                                loc, variance_scale, shape, scale)
        x_mode = loc
        sig2_mode = scale / (shape + 1)
        return x_mode, sig2_mode


normal_invgamma = normal_invgamma_gen()


class normal_invgamma_frozen(multi_rv_frozen):
    def __init__(self, loc=0, variance_scale=1, shape=1, scale=1, seed=None):

        self._dist = normal_invgamma_gen(seed)
        self.loc, self.variance_scale, self.shape, self.scale = self._dist._check_parameters(
                                                            loc, variance_scale, shape, scale)

    def logpdf(self, x, sig2):
        x, sig2 = self._dist._check_input(x, sig2)
        out = self._dist.logpdf(x, sig2, self.loc, self.variance_scale, self.shape, self.scale)
        return out

    def pdf(self, x, sig2):
        return np.exp(self.logpdf(x, sig2))

    def rvs(self, size=1, random_state=None):
        return self._dist.rvs(self.loc, self.variance_scale,
                            self.shape, self.scale, size, random_state)

    def mean(self):
        return self._dist.mean(self.loc, self.variance_scale, self.shape, self.scale)

    def mode(self):
        return self._dist.mode(self.loc, self.variance_scale, self.shape, self.scale)

# Set frozen generator docstrings from corresponding docstrings in
# normal_invgamma_gen and fill in default strings in class docstrings
for name in ['logpdf', 'pdf', 'rvs']:
    method = normal_invgamma_gen.__dict__[name]
    method_frozen = normal_invgamma_frozen.__dict__[name]
    #method_frozen.__doc__ = doccer.docformat(method.__doc__, nig_docdict_noparams)
    #method.__doc__ = doccer.docformat(method.__doc__, nig_docdict_params)
    
k=normal_invgamma.rvs(12,3,4,5, size=1000)
print (normal_invgamma.pdf(np.asarray([0]), sig2=np.asarray([1])))