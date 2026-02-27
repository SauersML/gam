# -*- coding: utf-8 -*-
from autograd.extend import primitive, defvjp
from autograd import numpy as np
from autograd.scipy.special import gammaln, beta
from autograd.numpy.numpy_vjps import unbroadcast_f
from scipy.special import (
    gammainc as _scipy_gammainc,
    gammaincc as _scipy_gammaincc,
    gamma,
    betainc as _scipy_betainc,
    gammainccinv as _scipy_gammainccinv,
    gammaincinv as _scipy_gammaincinv,
    betaincinv as _scipy_betaincinv,
)

__all__ = [
    "gamma",  # gamma function
    "gammainc",  # regularized lower incomplete gamma function
    "gammaincln",  # log of regularized lower incomplete gamma function
    "gammaincc",  # regularized upper incomplete gamma function
    "gammainccln",  # log of regularized upper incomplete gamma function
    "beta",  # beta function
    "betainc",  # incomplete beta function
    "betaincln",  # log of incomplete beta function
    'gammaincinv', # inverse of the gammainc w.r.t. the second argument, solves y = P(a, x)
    'gammainccinv', # inverse of the gammaincc w.r.t. the second argument, solves y = Q(a, x)
    'betaincinv', # inverse of the betainc, solves y = B(a, b, x)
]


LOG_EPISILON = 1e-35
MACHINE_EPISLON_POWER = np.finfo(float).eps ** (1 / 3)

gammainc = primitive(_scipy_gammainc)
gammaincc = primitive(_scipy_gammaincc)
betainc = primitive(_scipy_betainc)
gammainccinv = primitive(_scipy_gammainccinv)
gammaincinv = primitive(_scipy_gammaincinv)
betaincinv = primitive(_scipy_betaincinv)


@primitive
def gammainccln(a, x):
    return np.log(np.clip(gammaincc(a, x), LOG_EPISILON, 1 - LOG_EPISILON))


@primitive
def gammaincln(a, x):
    return np.log(np.clip(gammainc(a, x), LOG_EPISILON, 1 - LOG_EPISILON))


@primitive
def betaincln(a, b, x):
    return np.log(np.clip(betainc(a, b, x), LOG_EPISILON, 1 - LOG_EPISILON))


def central_difference_of_(f, argnum=0):
    """
    5th order approximation.
    """
    new_f = lambda x, *args: f(*args[:argnum], x, *args[argnum:])

    def _central_difference(_, *args):
        x = args[argnum]
        args = args[:argnum] + args[argnum + 1 :]

        # Why do we calculate a * MACHINE_EPSILON_POWER?
        # consider if x is massive, like, 2**100. Then even for a simple
        # function like the identity function, (2**100 + h) - 2**100 = 0  due
        # to floating points. (the correct answer should be 1.0)
        delta = np.maximum(x * MACHINE_EPISLON_POWER, 1e-7)

        # another thing to consider is that x is machine representable, but x + h is
        # rarely, and will be rounded to be machine representable. This (x + h) - x != h.
        temp = x + delta
        delta = temp - x
        return unbroadcast_f(
            x,
            lambda g: g
            * (
                -1 * new_f(x + 2 * delta, *args)
                + 8 * new_f(x + delta, *args)
                - 8 * new_f(x - delta, *args)
                + 1 * new_f(x - 2 * delta, *args)
            )
            / (12 * delta),
        )

    return _central_difference


def central_difference_of_log(f, argnum=0):
    """
    5th order approximation of derivative of log(f). We take advantage of the fact:

    d(log(f))/dx = 1/f df/dx

    So we approximate the second term only.
    """
    new_f = lambda x, *args: f(*args[:argnum], x, *args[argnum:])

    def _central_difference(_, *args):
        x = args[argnum]
        args = args[:argnum] + args[argnum + 1 :]

        # Why do we calculate a * MACHINE_EPSILON_POWER?
        # consider if x is massive, like, 2**100. Then even for a simple
        # function like the identity function, ((2**100 + h) - 2**100)/h = 0  due
        # to floating points. (the correct answer should be 1.0)
        delta = np.maximum(x * MACHINE_EPISLON_POWER, 1e-7)

        # another thing to consider is that x is machine representable, but x + h is
        # rarely, and will be rounded to be machine representable. This (x + h) - x != h.
        temp = x + delta
        delta = temp - x
        return unbroadcast_f(
            x,
            lambda g: g
            * (
                -1 * new_f(x + 2 * delta, *args)
                + 8 * new_f(x + 1 * delta, *args)
                - 8 * new_f(x - 1 * delta, *args)
                + 1 * new_f(x - 2 * delta, *args)
            )
            / (12 * delta)
            / new_f(x, *args),
        )

    return _central_difference


defvjp(
    gammainc,
    central_difference_of_(gammainc),
    lambda ans, a, x: unbroadcast_f(
        x, lambda g: g * np.exp(-x + np.log(x) * (a - 1) - gammaln(a))
    ),
)

defvjp(
    gammaincc,
    central_difference_of_(gammaincc),
    lambda ans, a, x: unbroadcast_f(
        x, lambda g: -g * np.exp(-x + np.log(x) * (a - 1) - gammaln(a))
    ),
)


defvjp(
    gammaincinv,
    central_difference_of_(gammaincinv),
    lambda ans, a, y: unbroadcast_f(
        y, lambda g: g * np.exp(gammaincinv(a, y) - np.log(gammaincinv(a, y)) * (a - 1) + gammaln(a))
    ),
)

defvjp(
    gammainccinv,
    central_difference_of_(gammainccinv),
    lambda ans, a, y: unbroadcast_f(
        y, lambda g: -g * np.exp(gammainccinv(a, y) - np.log(gammainccinv(a, y)) * (a - 1) + gammaln(a))
    ),
)

defvjp(
    gammainccln,
    central_difference_of_log(gammaincc),
    lambda ans, a, x: unbroadcast_f(
        x,
        lambda g: -g
        * np.exp(-x + np.log(x) * (a - 1) - gammaln(a) - gammainccln(a, x)),
    ),
)

defvjp(
    gammaincln,
    central_difference_of_log(gammainc),
    lambda ans, a, x: unbroadcast_f(
        x,
        lambda g: g * np.exp(-x + np.log(x) * (a - 1) - gammaln(a) - gammaincln(a, x)),
    ),
)


defvjp(
    betainc,
    central_difference_of_(betainc, argnum=0),
    central_difference_of_(betainc, argnum=1),
    lambda ans, a, b, x: unbroadcast_f(
        x, lambda g: g * np.power(x, a - 1) * np.power(1 - x, b - 1) / beta(a, b)
    ),
)

defvjp(
    betaincinv,
    central_difference_of_(betaincinv, argnum=0),
    central_difference_of_(betaincinv, argnum=1),
    lambda ans, a, b, y: unbroadcast_f(
        y, lambda g: g * 1/(np.power(betaincinv(a,b,y), a - 1) * np.power(1 - betaincinv(a,b,y), b - 1) / beta(a, b))
    ),
)


defvjp(
    betaincln,
    central_difference_of_log(betainc, argnum=0),
    central_difference_of_log(betainc, argnum=1),
    lambda ans, a, b, x: unbroadcast_f(
        x,
        lambda g: g
        * np.power(x, a - 1)
        * np.power(1 - x, b - 1)
        / beta(a, b)
        / betainc(a, b, x),
    ),
)
