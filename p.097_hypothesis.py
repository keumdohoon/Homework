#p.97_hypothesis

from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
    """Binomial(n, p)에 해당되는 mu(평균)와 sigma(표준편차)계산"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

from scratch.probability import normal_cdf

# 누적 분포 함수는 확률 변수가 특정 값보다 작을 확률을 나타낸다
normal_probability_below = normal_cdf

# 만약 확률 변수가 특정 값보다 적지 않으면, 특정 값도다 크다는 것을 의미한다. 
def normal_probability_above(lo: float,
                             mu: float = 0,
                             sigma: float = 1) -> float:
    """N(mu, sigma)를 따르는 정규 분포가 lo보다 클 확률을 나타내준다."""
    return 1 - normal_cdf(lo, mu, sigma)

def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# 만약 확률변수가 범위 밖에 존재한다면 범위 안에 존재하지 않다는 것을 의미한다. 
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
    """mu(평균)과 sigma(표준편차)를 따르는 정규분포가 lo와hi 사이에 없을 확률을 나타낸다"""
    return 1 - normal_probability_between(lo, hi, mu, sigma)

from scratch.probability import inverse_normal_cdf

def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """ P(Z <= z) = probability 인 z값을 반환"""
    return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
    """ P(Z >= z) = probability 인 z값을 반환"""
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                            mu: float = 0,
                            sigma: float = 1) -> Tuple[float, float]:
    """
    입력한 probability값을 포함하고
    평균을 중심으로 대칭적인 구간을 반환
    """
    tail_probability = (1 - probability) / 2

    # 구간의 상한은 tail_probability 값 이상의 확률 값을 가지고 있다.
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)

    #구간의 하한은 tail_probability 값 이하의 확률 값을 가지고 있다.

    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)


assert mu_0 == 500
assert 15.8 < sigma_0 < 15.9

# (469, 531)
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)


assert 468.5 < lower_bound < 469.5
assert 530.5 < upper_bound < 531.5

# p가 0.5 라고 가정할때, 유의 수준이 5%인구간
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# p = 0.55인 경우의 실제 평균과 표준편차
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# 제 2중 오류란 귀무가설(H0)을 기각하지 못한다는 의미
# 즉 X가 주어진 구간 안에 존재할 경우를 의미
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability      # 0.887


assert 0.886 < power < 0.888

hi = normal_upper_bound(0.95, mu_0, sigma_0)
# 결과값 526 (< 531,분포 상위 부분에 더 높은 확률을 주기 위해서)

type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability      # 0.936


assert 526 < hi < 526.1
assert 0.9363 < power < 0.9364
