using Distributions, PDMats, Primes, Random, LinearAlgebra

function qsimvnv(Σ::AbstractMatrix, a, b;m=nothing)
    #= rev 1.13
    
    This function uses an algorithm given in the paper
    "Numerical Computation of Multivariate Normal Probabilities", in
     J. of Computational and Graphical Stat., 1(1992), pp. 141-149, by
    Alan Genz, WSU Math, PO Box 643113, Pullman, WA 99164-3113
    Email : alangenz@wsu.edu
    The primary references for the numerical integration are
    "On a Number-Theoretical Integration Method"
    H. Niederreiter, Aequationes Mathematicae, 8(1972), pp. 304-11, and
    "Randomization of Number Theoretic Methods for Multiple Integration"
    R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13(1976), pp. 904-14.
    
    Re-coded in Julia from the MATLAB function qsimvnv(m,r,a,b)
    
    Alan Genz is the author the MATLAB qsimvnv() function.
    Alan Genz software website: http://archive.is/jdeRh
    Source code to MATLAB qsimvnv() function: http://archive.is/h5L37
    % QSIMVNV(m,r,a,b) and _chlrdr(r,a,b)
    %
    % Copyright (C) 2013, Alan Genz,  All rights reserved.
    %
    % Redistribution and use in source and binary forms, with or without
    % modification, are permitted provided the following conditions are met:
    %   1. Redistributions of source code must retain the above copyright
    %      notice, this list of conditions and the following disclaimer.
    %   2. Redistributions in binary form must reproduce the above copyright
    %      notice, this list of conditions and the following disclaimer in
    %      the documentation and/or other materials provided with the
    %      distribution.
    %   3. The contributor name(s) may not be used to endorse or promote
    %      products derived from this software without specific prior
    %      written permission.
    % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    % "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    % LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    % FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    % COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    % INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    % BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
    % OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    % ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
    % TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
    % OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    %
    
    Julia dependencies
    Distributions
    PDMats
    Primes
    Random
    LinearAlgebra =#
    
    if isnothing(m)
        m = 1000 * size(Σ, 1)  # default is 1000 * dimension
    end
    
    # check for proper dimensions
    n = size(Σ, 1)
    nc = size(Σ, 2) 	# assume square Cov matrix nxn
    # check dimension > 1
    n >= 2   || throw(ErrorException("dimension of Σ must be 2 or greater. Σ dimension: $(size(Σ))"))
    n == nc  || throw(DimensionMismatch("Σ matrix must be square. Σ dimension: $(size(Σ))"))
    
    # check dimensions of lower vector, upper vector, and cov matrix match
    (n == size(a, 1) == size(b, 1)) || throw(DimensionMismatch("inconsistent argument dimensions. Sizes: Σ $(size(Σ))  a $(size(a))  b $(size(b))"))
    
    # check that a and b are column vectors; if row vectors, fix it
    if size(a, 1) < size(a, 2)
        a = transpose(a)
    end
    if size(b, 1) < size(b, 2)
        b = transpose(b)
    end
    
    # check that lower integration limit a < upper integration limit b for all elements
    all(a .<= b) || throw(ArgumentError("lower integration limit a must be <= upper integration limit b"))
    
    # check that Σ is positive definate; if not, print warning
    isa(Σ, AbstractPDMat) || isposdef(Σ) || @warn "covariance matrix Σ fails positive definite check"
    
    # check if Σ, a, or b contains NaNs
    if any(isnan.(Σ)) || any(isnan.(a)) || any(isnan.(b))
        p = NaN
        e = NaN
        return (p, e)
    end
    
    # check if a==b
    if a == b
        p = 0.0
        e = 0.0
        return (p, e)
    end
    
    # check if a = -Inf & b = +Inf
    if all(a .== -Inf) && all(b .== Inf)
        p = 1.0
        e = 0.0
        return (p, e)
    end
    
    # check input Σ, a, b are floats; otherwise, convert them
    if eltype(Σ) <: Signed
        Σ = float(Σ)
    end
    
    if eltype(a) <: Signed
        a = float(a)
    end
    
    if eltype(b) <: Signed
        b = float(b)
    end
    
    ##################################################################
    #
    # Special cases: positive Orthant probabilities for 2- and
    # 3-dimesional Σ have exact solutions. Integration range [0,∞]
    #
    ##################################################################
    
    if all(a .== zero(eltype(a))) && all(b .== Inf) && n <= 3
        Σstd = sqrt.(diag(Σ))
        Rcorr = cov2cor(Σ, Σstd)
    
        if n == 2
            p = 1 / 4 + asin(Rcorr[1,2]) / (2π)
            e = eps()
        elseif n == 3
            p = 1 / 8 + (asin(Rcorr[1,2]) + asin(Rcorr[2,3]) + asin(Rcorr[1,3])) / (4π)
            e = eps()
        end
    
        return (p, e)
    
    end
    
    ##################################################################
    #
    # get lower cholesky matrix and (potentially) re-ordered integration vectors
    #
    ##################################################################
    
    (ch, as, bs) = _chlrdr(Σ, a, b) # ch =lower cholesky; as=lower vec; bs=upper vec
    
    ##################################################################
    #
    # quasi-Monte Carlo integration of MVN integral
    #
    ##################################################################
    
    ### setup initial values
    ai = as[1]
    bi = bs[1]
    ct = ch[1,1]
    
    unitnorm = Normal() # unit normal distribution
    rng = RandomDevice()
    
    # if ai is -infinity, explicity set c=0
    # implicitly, the algorith classifies anythign > 9 std. deviations as infinity
    if ai > -9 * ct
        if ai < 9 * ct
            c1 = cdf.(unitnorm, ai / ct)
        else
            c1 = 1.0
        end
    else
        c1 = 0.0
    end
    
    # if bi is +infinity, explicity set d=0
    if bi > -9 * ct
        if bi < 9 * ct
            d1 = cdf(unitnorm, bi / ct)
        else
            d1 = 1.0
        end
    else
        d1 = 0.0
    end
    
    # n=size(Σ,1) 	# assume square Cov matrix nxn
    cxi = c1			# initial cxi; genz uses ci but it conflicts with Lin. Alg. ci variable
    dci = d1 - cxi		# initial dcxi
    p = 0.0			# probablity = 0
    e = 0.0			# error = 0
    
    # Richtmyer generators
    ps = sqrt.(primes(Int(floor(5 * n * log(n + 1) / 4)))) # Richtmyer generators
    q = ps[1:n - 1,1]
    ns = 12
    nv = Int(max(floor(m / ns), 1))
    
    Jnv    = ones(1, nv)
    cfill  = transpose(fill(cxi, nv)) 	# evaulate at nv quasirandom points row vec
    dpfill = transpose(fill(dci, nv))
    y      = zeros(n - 1, nv)				# n-1 rows, nv columns, preset to zero
    
    #= Randomization loop for ns samples
     j is the number of samples to integrate over,
         but each with a vector nv in length
     i is the number of dimensions, or integrals to comptue =#
    
    for j in 1:ns					# loop for ns samples
        c  = copy(cfill)
        dc = copy(dpfill)
        pv = copy(dpfill)
        for i in 2:n
            x = transpose(abs.(2.0 .* mod.((1:nv) .* q[i - 1] .+ rand(rng), 1) .- 1))	 # periodizing transformation
                # note: the rand() is not broadcast -- it's a single random uniform value added to all elements
            y[i - 1,:] = quantile.(unitnorm, c .+ x .* dc)
            s = transpose(ch[i,1:i - 1]) * y[1:i - 1,:]
            ct = ch[i,i]										# ch is cholesky matrix
            ai = as[i] .- s
            bi = bs[i] .- s
            c = copy(Jnv)										# preset to 1 (>9 sd, +∞)
            d = copy(Jnv)										# preset to 1 (>9 sd, +∞)
    
            c[findall(x -> isless(x, -9 * ct), ai)] .= 0.0		# If < -9 sd (-∞), set to zero
            d[findall(x -> isless(x, -9 * ct), bi)] .= 0.0		# if < -9 sd (-∞), set to zero
            tstl = findall(x -> isless(abs(x), 9 * ct), ai)		# find cols inbetween -9 and +9 sd (-∞ to +∞)
            c[tstl] .= cdf.(unitnorm, ai[tstl] / ct)			# for those, compute Normal CDF
            tstl = (findall(x -> isless(abs(x), 9 * ct), bi))	# find cols inbetween -9 and +9 sd (-∞ to +∞)
            d[tstl] .= cdf.(unitnorm, bi[tstl] / ct)
            @. dc = d - c
            @. pv = pv * dc
        end # for i=
        d = (mean(pv) - p) / j
        p += d
        e = (j - 2) * e / j + d^2
    end # for j=
    
    e = 3 * sqrt(e) 	# error estimate is 3 times standard error with ns samples
    
    return (p, e)  	# return probability value and error estimate
end # function qsimvnv

function _chlrdr(Σ, a, b)

# Rev 1.13

# define constants
# 64 bit machien error 1.0842021724855e-19 ???
# 32 bit machine error 2.220446049250313e-16 ???
    ep = 1e-10 # singularity tolerance
    if Sys.WORD_SIZE == 64
        fpsize = Float64
        ϵ = eps(0.0) # 64-bit machine error
    else
        fpsize = Float32
        ϵ = eps(0.0f0) # 32-bit machine error
    end

    if !@isdefined sqrt2π
        sqrt2π = √(2π)
    end

# unit normal distribution
    unitnorm = Normal()

    n = size(Σ, 1) # covariance matrix n x n square

    ckk = 0.0
    dem = 0.0
    am = 0.0
    bm = 0.0
    ik = 0.0

    if eltype(Σ) <: Signed
        c = copy(float(Σ))
    else
        c = copy(Σ)
    end

    if eltype(a) <: Signed
        ap = copy(float(a))
    else
        ap = copy(a)
    end

    if eltype(b) <: Signed
        bp = copy(float(b))
    else
        bp = copy(b)
    end

    d = sqrt.(diag(c))
    for i in 1:n
        if d[i] > 0.0
            c[:,i] /= d[i]
            c[i,:] /= d[i]
            ap[i] = ap[i] / d[i]     # ap n x 1 vector
            bp[i] = bp[i] / d[i]     # bp n x 1 vector
        end
    end

    y = zeros(fpsize, n) # n x 1 zero vector to start

    for k in 1:n
        ik = k
        ckk = 0.0
        dem = 1.0
        s = 0.0
    # pprinta(c)
        for i in k:n
            if c[i,i] > ϵ  # machine error
                cii = sqrt(max(c[i,i], 0))

                if i > 1 && k > 1
                    s = (c[i,1:(k - 1)] .* y[1:(k - 1)])[1]
                end

                ai = (ap[i] - s) / cii
                bi = (bp[i] - s) / cii
                de = cdf(unitnorm, bi) - cdf(unitnorm, ai)

                if de <= dem
                    ckk = cii
                    dem = de
                    am = ai
                    bm = bi
                    ik = i
                end
            end # if c[i,i]> ϵ
        end # for i=
        i = n

        if ik > k
            ap[ik], ap[k] = ap[k], ap[ik]
            bp[ik], bp[k] = bp[k], bp[ik]

            c[ik,ik] = c[k,k]

            if k > 1
                c[ik,1:(k - 1)], c[k,1:(k - 1)] = c[k,1:(k - 1)], c[ik,1:(k - 1)]
            end

            if ik < n
                c[(ik + 1):n,ik], c[(ik + 1):n,k] = c[(ik + 1):n,k], c[(ik + 1):n,ik]
            end

            if k <= (n - 1) && ik <= n
                c[(k + 1):(ik - 1),k], c[ik,(k + 1):(ik - 1)] = transpose(c[ik,(k + 1):(ik - 1)]), transpose(c[(k + 1):(ik - 1),k])
            end
        end # if ik>k

        if ckk > k * ep
            c[k,k] = ckk
            if k < n
                c[k:k,(k + 1):n] .= 0.0
            end

            for i in (k + 1):n
                c[i,k] /= ckk
                c[i:i,(k + 1):i] -= c[i,k] * transpose(c[(k + 1):i,k])
            end

            if abs(dem) > ep
                y[k] = (exp(-am^2 / 2) - exp(-bm^2 / 2)) / (sqrt2π * dem)
            else
                if am < -10
                    y[k] = bm
                elseif bm > 10
                y[k] = am
            else
                y[k] = (am + bm) / 2
                end
            end # if abs
        else
            c[k:n,k] .== 0.0
            y[k] = 0.0
        end # if ckk>ep*k
    end # for k=

    return (c, ap, bp)
end # function _chlrdr

function testmvn(;m=nothing)

# Typical Usage/Example Code
# Example multivariate Normal CDF for various vectors
    println()

# from MATLAB documentation
    r = [4 3 2 1;3 5 -1 1;2 -1 4 2;1 1 2 5]
 	a = [-Inf; -Inf; -Inf; -Inf] # -inf for each
	b = [1; 2; 3; 4 ]
	m = 5000
	# m=4000 # rule of thumb: 1000*(number of variables)
	(myp, mye) = qsimvnv(r, a, b;m=m)
	println("Answer should about 0.6044 to 0.6062")
	println(myp)
	println("The Error should be <= 0.001 - 0.0014");
	println(mye)

	r = [1 0.6 0.333333;
	   0.6 1 0.733333;
	   0.333333 0.733333 1]
	r  = [  1   3 / 5   1 / 3;
		  3 / 5    1    11 / 15;
		  1 / 3  11 / 15    1]
	a = [-Inf;-Inf;-Inf]
	b = [1;4;2]
	# m=3000;
	(myp, mye) = qsimvnv(r, a, b;m=4000)
	println()
	println("Answer shoudl be about 0.82798")
	# answer from Genz paper 0.827984897456834
	println(myp)
	println("The Error should be <= 2.5-05")
	println(mye)

	r = [1 0.25 0.2; 0.25 1 0.333333333; 0.2 0.333333333 1]
	a = [-1;-4;-2]
	b = [1;4;2]
	# m=3000;
	(myp, mye) = qsimvnv(r, a, b;m=4000);
	println()
	println("Answer should be about 0.6537")
	println(myp)
	println("The Error should be <= 2.5-05")
	println(mye)

	# Genz problem 1.5 p. 4-5  & p. 63
	# correct answer F(a,b) = 0.82798
    r = [1 / 3 3 / 5 1 / 3;
         3 / 5 1.0 11 / 15;
         1 / 3 11 / 15 1.0]
    a = [-Inf; -Inf; -Inf]
    b = [1; 4; 2]
	(myp, mye) = qsimvnv(r, a, b;m=4000)
	println()
	# println("Answer shoudl be about 0.82798")
	println("Answer should be 0.9432")
	println(myp)
	println("The error should be < 6e-05")
	println(mye)

	# Genz page 63 uses a different r Matrix
	r = [1 0 0;
		3 / 5 1 0;
		1 / 3 11 / 15 1]
		a = [-Inf; -Inf; -Inf]
		b = [1; 4; 2]
	(myp, mye) = qsimvnv(r, a, b;m=4000)
	println()
	println("Answer shoudl be about 0.82798")
	println(myp)
	println("The error should be < 6e-05")
	println(mye)
	# mystery solved - he used the wrong sigma Matrix
	# when computing the results on page 6


	# singular cov Example
	r = [1 1 1; 1 1 1; 1 1 1]
	a = [-Inf, -Inf, -Inf]
	b = [1, 1, 1]
	(myp, mye) = qsimvnv(r, a, b)
	println()
	println("Answer should be 0.841344746068543")
	println(myp)
	println("The error should be 0.0")
	println(mye)
	println("Cov matrix is singular")
	println("Problem reduces to a univariate problem with")
	println("p = cdf.(Normal(),1) = ", cdf.(Normal(), 1))

	# 5 dimensional Example
	# c = LinearAlgebra.tri(5)
	r = [1 1 1 1 1;
		 1 2 2 2 2;
		 1 2 3 3 3;
		 1 2 3 4 4;
		 1 2 3 4 5]
	a = [-1,-2,-3,-4,-5]
	b = [2,3,4,5,6]
	(myp, mye) = qsimvnv(r, a, b)
	println()
	println("Answer should be ~ 0.7613")
	# genz gives 0.4741284  p. 5 of book
	# Julia, MATLAB, and R all give ~0.7613 !
	println(myp)
	println("The error should be < 0.001")
	println(mye)

	# genz reversed the integration limits when computing
	# see p. 63
	a = sort!(a)
	b = 1 .- a
	(myp, mye) = qsimvnv(r, a, b)
	println()
	println("Answer should be ~ 0.4741284")
	# genz gives 0.4741284  p. 5 of book
	# now matches with reversed integration limits
	println(myp)
	println("The error should be < 0.001")
	println(mye)

	# positive orthant of above
	a = [0,0,0,0,0]
	(myp, mye) = qsimvnv(r, a, b)
	println()
	println("Answer should be ~  0.11353418")
	# genz gives 0.11353418   p. 6 of book
	println(myp)
	println("The error should be < 0.001")
	println(mye)

	# now with a = -inf
	a = [-Inf,-Inf,-Inf,-Inf,-Inf]
	(myp, mye) = qsimvnv(r, a, b)
	println()
	println("Answer should be ~ 0.81031455")
	# genz gives 0.81031455  p. 6 of book
	println(myp)
	println("The error should be < 0.001")
	println(mye)

	# eight dimensional Example
	r = [1 1 1 1 1 1 1 1;
		 1 2 2 2 2 2 2 2;
		 1 2 3 3 3 3 3 3;
		 1 2 3 4 4 4 4 4;
		 1 2 3 4 5 5 5 5;
		 1 2 3 4 5 6 6 6;
		 1 2 3 4 5 6 7 7;
		 1 2 3 4 5 6 7 8]
	a = -1 * [1,2,3,4,5,6,7,8]
	b = [2,3,4,5,6,7,8,9]
	(myp, mye) = qsimvnv(r, a, b)
	println()
	println("Answer should be ~ 0.7594")
	# genz gives 0.32395   p. 6 of book
	# MATLAB gives 0.7594362
	println(myp)
	println("The error should be < 0.001")
	println(mye)


	# eight dim orthant
	a = [0,0,0,0,0,0,0,0]
	b = [Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf]
	(myp, mye) = qsimvnv(r, a, b)
	println()
	println("Answer should be ~ 0.196396")
	# genz gives 0.076586  p. 6 of book
	# MATLB gives 0.196383
	println(myp)
	println("The error should be < 0.001")
	println(mye)

	# eight dim with a = -inf
	a = -Inf * [1,1,1,1,1,1,1,1]
	b = [2,3,4,5,6,7,8,9]
	# b = [0,0,0,0,0,0,0,0]
	(myp, mye) = qsimvnv(r, a, b)
	println()
	println("Answer should be ~ 0.9587")
	# genz gives 0.69675    p. 6 of book
	# MATLAB gives 0.9587
	println(myp)
	println("The error should be < 0.001")
	println(mye)
end # testmvn
