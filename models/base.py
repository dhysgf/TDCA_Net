

from typing import Optional, List, Tuple, Union
import warnings
import numpy as np
from numpy import ndarray
from scipy.linalg import solve
from scipy.signal import sosfiltfilt, cheby1, cheb1ord
from sklearn.base import BaseEstimator, TransformerMixin, clone
import mne


def robust_pattern(W : ndarray, Cx: ndarray, Cs: ndarray) -> ndarray:
    """Transform spatial filters to spatial patterns based on paper [1]_.
        Referring to the method mentioned in article [1],the constructed spatial filter only shows how to combine
        information from different channels to extract signals of interest from EEG signals, but if our goal is
        neurophysiological interpretation or visualization of weights, activation patterns need to be constructed
        from the obtained spatial filters.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    W : ndarray
        Spatial filters, shape (n_channels, n_filters).
    Cx : ndarray
        Covariance matrix of eeg data, shape (n_channels, n_channels).
    Cs : ndarray
        Covariance matrix of source data, shape (n_channels, n_channels).

    Returns
    -------
    A : ndarray
        Spatial patterns, shape (n_channels, n_patterns), each column is a spatial pattern.

    References
    ----------
    .. [1] Haufe, Stefan, et al. "On the interpretation of weight vectors of linear models in multivariate neuroimaging.
           Neuroimage 87 (2014): 96-110.
    """
    # use linalg.solve instead of inv, makes it more stable
    # see https://github.com/robintibor/fbcsp/blob/master/fbcsp/signalproc.py
    # and https://ww2.mathworks.cn/help/matlab/ref/mldivide.html
    A = solve(Cs.T, np.dot(Cx, W).T).T
    return A


class FilterBank(BaseEstimator, TransformerMixin):
    """
    Filter bank decomposition is a bandpass filter array that divides the input signal into
    multiple subband components and obtains the eigenvalues of each subband component.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    base_estimator : class
        Estimator for model training and feature extraction.
    filterbank : list[ndarray]
        A bandpass filter bank used to divide the input signal into multiple subband components.
    n_jobs : int
        Sets the number of CPU working cores. The default is None.

    References
    ----------
    .. [1] Chen X, Wang Y, Nakanishi M, et al. High-speed spelling with a noninvasive brain-computer interface[J].
    Proceedings of the national academy of sciences, 2015, 112(44): E6058-E6067.
    """
    def __init__(
        self,
        base_estimator: BaseEstimator,
        filterbank: List[ndarray],
        n_jobs: Optional[int] = None,
    ):
        self.base_estimator = base_estimator
        self.filterbank = filterbank
        self.n_jobs = n_jobs

    def fit(self, X: ndarray, y: Optional[ndarray] = None, **kwargs):
        """
        Training model

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        X : None
            Training signal (parameters can be ignored, only used to maintain code structure).
        y : None
            Label data (ibid., ignorable).
        Yf : None
            Reference signal (ibid., ignorable).
        """
        self.estimators_ = [
            clone(self.base_estimator) for _ in range(len(self.filterbank))
        ]
        X = self.transform_filterbank(X)
        for i, est in enumerate(self.estimators_):
            est.fit(X[i], y, **kwargs)
        # def wrapper(est, X, y, kwargs):
        #     est.fit(X, y, **kwargs)
        #     return est
        # self.estimators_ = Parallel(n_jobs=self.n_jobs)(
        #     delayed(wrapper)(est, X[i], y, kwargs) for i, est in enumerate(self.estimators_))
        return self

    def transform(self, X: ndarray, **kwargs):
        """
        The parameters stored in self are used to convert X into features, and X is filtered through the filter bank to
        obtain the eigenvalues of each subband component.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        X : ndarray, shape(n_trials, n_channels, n_samples)
            Test the signal.

        Returns
        -------
        feat : ndarray, shape(n_trials, n_fre)
            Feature array.
        """
        X = self.transform_filterbank(X)
        feat = [est.transform(X[i], **kwargs) for i, est in enumerate(self.estimators_)]
        # def wrapper(est, X, kwargs):
        #     retval = est.transform(X, **kwargs)
        #     return retval
        # feat = Parallel(n_jobs=self.n_jobs)(
        #     delayed(wrapper)(est, X[i], kwargs) for i, est in enumerate(self.estimators_))
        feat = np.concatenate(feat, axis=-1)
        return feat

    def transform_filterbank(self, X: ndarray):
        """
        The input signal is filtered through a filter bank.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        X : ndarray, shape(n_trials, n_channels, n_samples)
            Input signal.

        Returns
        -------
        Xs: ndarray, shape(Nfb, n_trials, n_channels, n_samples)
            Individual subband components of the input signal.
        """
        Xs = np.stack([sosfiltfilt(sos, X, axis=-1) for sos in self.filterbank])
        return Xs


class FilterBankSSVEP(FilterBank):
    """
    Filter bank analysis for SSVEP.
    The SSVEP is analyzed using filter banks, that is, multiple filters are combined to decompose the SSVEP signal
    into specific segments (subbands containing the original data) and obtain its characteristic data.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    filterbank : list[ndarray]
        The filter bank.
    base_estimator : class
        Estimator for model training and feature extraction.
    filterweights : ndarray
        Filter weight, default is None.
    n_jobs : int
        Sets the number of CPU working cores. The default is None.
    """

    def __init__(
        self,
        filterbank: List[ndarray],
        base_estimator: BaseEstimator,
        filterweights: Optional[ndarray] = None,
        n_jobs: Optional[int] = None,
    ):
        self.filterweights = filterweights
        super().__init__(base_estimator, filterbank, n_jobs=n_jobs)

    def transform(self, X: ndarray):  # type: ignore[override]
        """
        X is converted into features by using the parameters stored in self, and the eigenvalues of each subband
        component are obtained after the input signal is filtered by the filter bank.

        update log:
            2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

        Parameters
        ----------
        X : ndarray, shape(n_trials, n_channels, n_samples)
            Test the signal.

        Returns
        -------
        features : ndarray, shape(n_trials, n_fre)
            Feature array.
        """
        features = super().transform(X)#顶类做了transfomer
        if self.filterweights is None:
            return features
        else:
            features = np.reshape(
                features, (features.shape[0], len(self.filterbank), -1)
            )
            return np.sum(
                features * self.filterweights[np.newaxis, :, np.newaxis], axis=1
            )


def generate_filterbank(
    passbands: List[Tuple[float, float]],
    stopbands: List[Tuple[float, float]],
    srate: int,
    order: Optional[int] = None,
    rp: float = 0.5,
):
    """
    Create a filter bank, that is, obtain a bandpass filter coefficient that can divide the input signal into multiple
    subband components.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    passbands : list or tuple(float, float)
        Passband parameters.
    stopbands : list or tuple(float, float)
        Stopband parameters.
    srate : float
        Sampling rate.
    order : int
        Filter order.
    rp : float
        The maximum ripple allowed in the passband below the unit gain is 0.5 by default.

    Returns
    -------
    Filterbank：ndarray, shape(len(passbands), N, 6)
        Filter bank coefficient.
    """
    filterbank = []
    for wp, ws in zip(passbands, stopbands):
        if order is None:
            N, wn = cheb1ord(wp, ws, 3, 40, fs=srate)
            sos = cheby1(N, rp, wn, btype="bandpass", output="sos", fs=srate)
        else:
            sos = cheby1(order, rp, wp, btype="bandpass", output="sos", fs=srate)

        filterbank.append(sos)
    return filterbank


def generate_cca_references(
    freqs: Union[ndarray, int, float],
    srate,
    T,
    phases: Optional[Union[ndarray, int, float]] = None,
    n_harmonics: int = 1,
):
    """
    Construct a sine-cosine reference signal for canonical correlation analysis (CCA).

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    freqs : int or float
        Frequency.
    srate : int
        Sampling rate.
    T : int
        Sampling time.
    phases : int or float
        Phase, default is None.
    n_harmonics : int
        The number of harmonics. The default value is 1.

    Returns
    -------
    Yf：ndarray, shape(srate*T, n_harmonics*2)
        Sine and cosine reference signal.
    """
    if isinstance(freqs, int) or isinstance(freqs, float):
        freqs = np.array([freqs])
    freqs = np.array(freqs)[:, np.newaxis]
    if phases is None:
        phases = 0
    if isinstance(phases, int) or isinstance(phases, float):
        phases = np.array([phases])
    phases = np.array(phases)[:, np.newaxis]
    t = np.linspace(0, T, int(T * srate))

    Yf = []
    for i in range(n_harmonics):
        Yf.append(
            np.stack(
                [
                    np.sin(2 * np.pi * (i + 1) * freqs * t + np.pi * phases),
                    np.cos(2 * np.pi * (i + 1) * freqs * t + np.pi * phases),
                ],
                axis=1,
            )
        )
    Yf = np.concatenate(Yf, axis=1)
    return Yf


def sign_flip(u, s, vh=None):
    """Flip signs of SVD or EIG using the method in paper [1]_.

    update log:
        2023-12-10 by Leyi Jia <18020095036@163.com>, Add code annotation

    Parameters
    ----------
    u: ndarray
        left singular vectors, shape (M, K).
    s: ndarray
        singular values, shape (K,).
    vh: ndarray or None
        transpose of right singular vectors, shape (K, N).

    Returns
    -------
    u: ndarray
        corrected left singular vectors.
    s: ndarray
        singular values.
    vh: ndarray
        transpose of corrected right singular vectors.

    References
    ----------
    .. [1] https://www.sandia.gov/~tgkolda/pubs/pubfiles/SAND2007-6422.pdf
    """
    if vh is None:
        total_proj = np.sum(u * s, axis=0)
        signs = np.sign(total_proj)

        random_idx = signs == 0
        if np.any(random_idx):
            signs[random_idx] = 1
            warnings.warn(
                "The magnitude is close to zero, the sign will become arbitrary."
            )
        u = u * signs
        return u, s
    else:
        left_proj = np.sum(s[:, np.newaxis] * vh, axis=-1)
        right_proj = np.sum(u * s, axis=0)
        total_proj = left_proj + right_proj
        signs = np.sign(total_proj)
        random_idx = signs == 0
        if np.any(random_idx):
            signs[random_idx] = 1
            warnings.warn(
                "The magnitude is close to zero, the sign will become arbitrary."
            )
        u = u * signs
        vh = signs[:, np.newaxis] * vh
        return u, s, vh
