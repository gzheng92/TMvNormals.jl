# TMvNormals.jl
Implements a truncated multivariate normal distribution.

Draws heavily on formulas from Manjunath Wilhem (2021)

## Usage

Builds on `Distributions.jl`

```
# Define a distribution
d = TMvNormal(
  zeros(3),
  ones((3,3)) + 5*I(3),
  [-1, -1, -Inf],
  [1, 2, 5]
)

rand(d, 5) # Produces a 5x3 matrix 

pdf(d, zeros(3)) # evaluates the joint density

let d = 1
  # evaluates the marginal univariate density along dimension d
  pdf(d, 0, 1)
end

let i=1, j=2
  # evalutes the marginal bivariate density along dimensions i and j
  pdf(d, zeros(2), [1, 2])
end

cdf(d, zeros(3)) # evaluates the cdf

# Also evaluates first and second order moments
mean(d)
cov(d)

# As in intermediary step, uses Genz (1992) algorithm
# to compute the CDF of a multivariate normal distribution

# the latent untruncated normal is contained in d.𝒩
cdf(𝒩, zeros(3))
```

## References

Manjunath, B. G. and Stefan Wilhelm. “Moments Calculation for the Doubly Truncated Multivariate Normal Density.” (2021).
