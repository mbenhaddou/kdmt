import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce

_log2 = lambda x: _math.log2(x)
_ln = _math.log

_product = lambda s: reduce(lambda x, y: x * y, s)

_SMALL = 1e-20

try:
    from scipy.stats import fisher_exact
except ImportError:
    pass

NGRAM = 0
"""Marginals index for the ngram count"""

UNIGRAMS = -2
"""Marginals index for a tuple of each unigram count"""

TOTAL = -1
"""Marginals index for the number of words in the data"""


def _expected_values(cont):
    """Calculates expected values for a contingency table."""
    n_xx = sum(cont)
    # For each contingency table cell
    for i in range(4):
        yield (cont[i] + cont[i ^ 1]) * (cont[i] + cont[i ^ 2]) / n_xx
        
def raw_freq(*marginals):
    """Scores ngrams by their frequency"""
    return marginals[NGRAM] / marginals[TOTAL]
    
def student_t(*marginals):
        """Scores ngrams using Student's t test with independence hypothesis
        for unigrams, as in Manning and Schutze 5.3.1.
        """
        return (
            marginals[NGRAM]
            - _product(marginals[UNIGRAMS]) / (marginals[TOTAL])
        ) / (marginals[NGRAM] + _SMALL) ** 0.5

    
def chi_sq(*marginals):
        """Scores ngrams using Pearson's chi-square as in Manning and Schutze
        5.3.3.
        """
        cont = _contingency(*marginals)
        exps = _expected_values(cont)
        return sum((obs - exp) ** 2 / (exp + _SMALL) for obs, exp in zip(cont, exps))


def mi_like(*marginals, **kwargs):
        """Scores ngrams using a variant of mutual information. The keyword
        argument power sets an exponent (default 3) for the numerator. No
        logarithm of the result is calculated.
        """
        return marginals[NGRAM] ** kwargs.get("power", 3) / _product(
            marginals[UNIGRAMS]
        )

    
def pmi( *marginals):
        """Scores ngrams by pointwise mutual information, as in Manning and
        Schutze 5.4.
        """
        return _log2(marginals[NGRAM] * marginals[TOTAL]) - _log2(
            _product(marginals[UNIGRAMS])
        )

    
def likelihood_ratio(*marginals):
        """Scores ngrams using likelihood ratios as in Manning and Schutze 5.3.4."""
        cont = _contingency(*marginals)
        return 2 * sum(
            obs * _ln(obs / (exp + _SMALL) + _SMALL)
            for obs, exp in zip(cont, _expected_values(cont))
        )

    
def poisson_stirling(cls, *marginals):
        """Scores ngrams using the Poisson-Stirling measure."""
        exp = _product(marginals[UNIGRAMS]) / (marginals[TOTAL] ** (cls._n - 1))
        return marginals[NGRAM] * (_log2(marginals[NGRAM] / exp) - 1)

    
def jaccard(cls, *marginals):
        """Scores ngrams using the Jaccard index."""
        cont = cls._contingency(*marginals)
        return cont[0] / sum(cont[:-1])



def _contingency(n_ii, n_ix_xi_tuple, n_xx):
        """Calculates values of a bigram contingency table from marginal values."""
        (n_ix, n_xi) = n_ix_xi_tuple
        n_oi = n_xi - n_ii
        n_io = n_ix - n_ii
        return (n_ii, n_oi, n_io, n_xx - n_ii - n_oi - n_io)

    
def _marginals(n_ii, n_oi, n_io, n_oo):
        """Calculates values of contingency table marginals from its values."""
        return (n_ii, (n_oi + n_ii, n_io + n_ii), n_oo + n_oi + n_io + n_ii)

    


    
def phi_sq(*marginals):
        """Scores bigrams using phi-square, the square of the Pearson correlation
        coefficient.
        """
        n_ii, n_io, n_oi, n_oo = _contingency(*marginals)

        return (n_ii * n_oo - n_io * n_oi) ** 2 / (
            (n_ii + n_io) * (n_ii + n_oi) * (n_io + n_oo) * (n_oi + n_oo)
        )

    
def chi_sq(n_ii, n_ix_xi_tuple, n_xx):
        """Scores bigrams using chi-square, i.e. phi-sq multiplied by the number
        of bigrams, as in Manning and Schutze 5.3.3.
        """
        (n_ix, n_xi) = n_ix_xi_tuple
        return n_xx * phi_sq(n_ii, (n_ix, n_xi), n_xx)

    
def fisher(*marginals):
        """Scores bigrams using Fisher's Exact Test (Pedersen 1996).  Less
        sensitive to small counts than PMI or Chi Sq, but also more expensive
        to compute. Requires scipy.
        """

        n_ii, n_io, n_oi, n_oo = _contingency(*marginals)

        (odds, pvalue) = fisher_exact([[n_ii, n_io], [n_oi, n_oo]], alternative="less")
        return pvalue

    
def dice(n_ii, n_ix_xi_tuple, n_xx):
        """Scores bigrams using Dice's coefficient."""
        (n_ix, n_xi) = n_ix_xi_tuple
        return 2 * n_ii / (n_ix + n_xi)
