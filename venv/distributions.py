from random import randrange
import math
import numpy as np
from typing import Tuple

MU = 'mu'
SIGMA = 's'
LAMBDA = 'lambda'
A = 'a'
B = 'b'


def get_distributions():
    ds = [Normal("Normal", {'mu': 0, 's': 1}),
          Cauchy("Cauchy", {'mu': 0, 'lambda': 1}),
          Laplace("Laplace", {'mu': 0, 'lambda': 2 ** (-0.5)}),
          Poisson("Poisson", {'mu': 10}),
          Uniform("Uniform", {'a': -3 ** 0.5, 'b': 3 ** 0.5})]
    return ds


def selection(dist, n: int):
    return sorted([dist.x() for i in range(n)])


def r() -> float:
    return randrange(1, 1000) / 1000


class AbstractDistribution:
    name: str
    parameters: dict

    def __init__(self, name: str, parameters: dict):
        self.name = name
        self.parameters = parameters

    def x(self) -> float:
        return 0

    def f(self, x: float) -> float:
        return 0

    def F(self, x: float) -> float:
        return 0

    def interval(self) -> Tuple[float, float]:
        return (0, 0)

    def discrete(self):
        return False


class Normal(AbstractDistribution):
    laplas_table = np.zeros(1)

    def f(self, x: float) -> float:
        s = self.parameters[SIGMA]
        m = self.parameters[MU]
        return math.exp(-(x - m) ** 2 / (2 * s)) / (s * math.sqrt(2 * math.pi))

    def x(self) -> float:
        y = -6
        for i in range(12):
            y += r()
        return self.parameters[MU] + self.parameters[SIGMA] * y

    def interval(self):
        s = self.parameters[SIGMA]
        m = self.parameters[MU]
        return (m - 4 * s, m + 4 * s)


class Cauchy(AbstractDistribution):
    def f(self, x: float) -> float:
        l = self.parameters[LAMBDA]
        m = self.parameters[MU]
        return l / (math.pi * (l ** 2 + (x - m) ** 2))

    def F(self, x: float) -> float:
        l = self.parameters[LAMBDA]
        m = self.parameters[MU]
        return 0.5 + math.atan((x - m) / l) / math.pi

    def x(self) -> float:
        e = 0.0001
        l = self.parameters[LAMBDA]
        m = self.parameters[MU]
        while True:
            y = r()
            if abs(y - 0.25) > e and abs(y - 0.75) > e:
                return m + l * math.tan(2 * math.pi * y)

    def interval(self) -> float:
        l = self.parameters[LAMBDA]
        m = self.parameters[MU]
        rng = math.sqrt(l * (300 - l))
        return (m - rng, m + rng)


class Laplace(AbstractDistribution):
    def f(self, x: float) -> float:
        l = self.parameters[LAMBDA]
        m = self.parameters[MU]
        return 0.5 * l * math.exp(-l * abs(x - m))

    def F(self, x: float) -> float:
        l = self.parameters[LAMBDA]
        m = self.parameters[MU]
        if x < m:
            return 0.5 * math.exp(l * (x - m))
        else:
            return 1 - 0.5 * math.exp(-l * (x - m))

    def x(self):
        l = self.parameters[LAMBDA]
        m = self.parameters[MU]
        return m + math.log(r() / r()) / l

    def interval(self):
        l = self.parameters[LAMBDA]
        m = self.parameters[MU]
        rng = -math.log(0.002 / l) / l
        return (m - rng, m + rng)


class Poisson(AbstractDistribution):
    def f(self, x: float) -> float:
        if x < 0:
            return 0
        m = self.parameters[MU]
        n = math.floor(x)
        v = math.exp(-m)
        for i in range(1, n + 1):
            v *= m / i
        return v

    def F(self, x: float) -> float:
        if x < 0:
            return 0
        m = self.parameters[MU]
        k = math.ceil(x)
        v = math.exp(-m)
        s = v
        for i in range(1, k):
            v *= m / i
            s += v
        return s

    def x(self) -> float:
        m = self.parameters[MU]
        p = math.exp(-m)
        r1 = r() - p
        x = 0
        while r1 > 0:
            x += 1
            p *= m / x
            r1 -= p
        return x

    def interval(self):
        return (0, self.parameters[MU] * 3)

    def discrete(self):
        return True


class Uniform(AbstractDistribution):
    def f(self, x: float) -> float:
        a = self.parameters[A]
        b = self.parameters[B]
        if x < a or x > b:
            return 0
        else:
            return 1 / (b - a)

    def F(self, x: float) -> float:
        a = self.parameters[A]
        b = self.parameters[B]
        if x < a:
            return 0
        elif x > b:
            return 1
        else:
            return (x - a) / (b - a)

    def x(self) -> float:
        a = self.parameters[A]
        b = self.parameters[B]
        return a + (b - a) * r()

    def interval(self):
        a = self.parameters[A]
        b = self.parameters[B]
        return (a, b)
