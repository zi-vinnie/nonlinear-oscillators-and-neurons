from typing import Callable


def euler_step_2d(
    x0: float,
    y0: float,
    dt: float,
    f: Callable[[float, float], float],
    g: Callable[[float, float], float],
) -> tuple[float, float]:
    """
    Euler time stepping method for solving 2D ODEs.
    """
    x = x0 + dt * f(x0, y0)
    y = y0 + dt * g(x0, y0)
    return x, y


def midpoint_step_2d(
    x0: float,
    y0: float,
    dt: float,
    f: Callable[[float, float], float],
    g: Callable[[float, float], float],
) -> tuple[float, float]:
    """
    Midpoint method for solving 2D ODEs.
    """
    x_mid = x0 + dt / 2 * f(x0, y0)
    y_mid = y0 + dt / 2 * g(x0, y0)
    x = x0 + dt * f(x_mid, y_mid)
    y = y0 + dt * g(x_mid, y_mid)
    return x, y


def dx_dtWithParams(I: float = 0) -> Callable[[float, float], float]:
    """
    Returns a dx/dt function with the given parameters
    """
    return lambda x, y: _dx_dt(x, y, I)


def dy_dtWithParams(
    mu: float, a: float = 0, b: float = 0
) -> Callable[[float, float], float]:
    """
    Returns a dy/dt function with the given parameters
    """
    return lambda x, y: _dy_dt(x, y, mu, a, b)


def _dx_dt(x: float, y: float, I: float) -> float:
    return x - 1 / 3 * x**3 - y + I


def _dy_dt(x: float, y: float, mu: float, a: float, b: float) -> float:
    return 1 / mu * (x - a * y + b)


class FitzHughNagumoModelParams:
    def __init__(self, I: float, mu: float, a: float, b: float):
        self.I = I
        self.mu = mu
        self.a = a
        self.b = b
