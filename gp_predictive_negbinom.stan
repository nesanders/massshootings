
### Base is Section 14.5 of Stan 2.8.0 manual, the logistic classification example
data {
	int<lower=1> N1;
	vector[N1] x1;
	int z1[N1];
	int<lower=1> N2;
	vector[N2] x2;
}
transformed data {
	int<lower=1> N;
	vector[N1+N2] x;
	N = N1 + N2;
	for (n in 1:N1) x[n] = x1[n];
	for (n in 1:N2) x[N1 + n] = x2[n];
}
parameters {
	vector[N1] y1;
	vector[N2] y2;
	real<lower=0> eta_sq;
	real<lower=1> inv_rho;
	real<lower=1e-6> sigma_sq;
	real mu_0;
	real mu_b;
	real<lower=0> NB_phi_inv;
}
#transformed parameters {
	#real<lower=0> rho_sq;
	
	#rho_sq = inv(inv_rho_sq);
	#}
model {
	matrix[N,N] Sigma;
	vector[N] y;
	matrix[N1+N2,N1+N2] L;
	vector[N1+N2] mu;
	
	// Calculate mean function
	for (i in 1:N) mu[i] = mu_0 + mu_b * x[i];

	// off-diagonal elements
	for (i in 1:(N-1)) {
		for (j in (i+1):N) {
			Sigma[i,j] = eta_sq * exp(-pow(inv_rho,-2) * pow(x[i] - x[j],2));
			Sigma[j,i] = Sigma[i,j];
		}
	}
	// diagonal elements
	for (k in 1:N)
		Sigma[k,k] = eta_sq + sigma_sq;
		#Sigma[k,k] = eta_sq;
	
	// Decompose
	L = cholesky_decompose(Sigma);
	
	// GP hyperpriors
	eta_sq ~ cauchy(0, 1);
	sigma_sq ~ cauchy(0, 1);
	inv_rho ~ gamma(4, 1); // Gamma prior with mean of 4 and std of 2
	
// Visualize with scipy
// from scipy.stats import gamma; import numpy as np; from itertools import permutations
// px = logspace(0,2, 4, base=2)
// plt.figure()
// for a,b in permutations(px, 2):
// 	mygamma = gamma(a, scale=b)
// 	x = np.linspace(mygamma.ppf(0.0001), mygamma.ppf(0.9999), 1000)
// 	plt.plot(x, mygamma.pdf(x), label="$\\alpha=%0.2f, \\beta=%0.2f$" %(a,b))
// plt.legend(prop={'size':7})
// 
// plt.figure()
// mygamma = gamma(4, scale=1)
// x = np.linspace(mygamma.ppf(0.0001), mygamma.ppf(0.9999), 1000)
// plt.plot(x, mygamma.pdf(x), label="$\\alpha=4, \\beta=1$")


	// Mean model priors
	mu_0 ~ normal(0, 2);
	mu_b ~ normal(0, 0.2);
	
	// Negative-binomial prior
	// For neg_binomial_2, phi^-1 controls the overdispersion.  phi^-1 ~ 0 reduces to the poisson.  phi^-1 = 1 represents variance = mu+mu^2
	NB_phi_inv ~ cauchy(0, 0.5);
	
	// Parameter sampling
	for (n in 1:N1) y[n] = y1[n];
	for (n in 1:N2) y[N1 + n] = y2[n];
	y ~ multi_normal_cholesky(mu, L);
	
	// Likelihood
	for (n in 1:N1) {
		z1[n] ~ neg_binomial_2(exp(y1[n]), inv(NB_phi_inv));
	}
}