import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.optimize import differential_evolution

def calculate_log_returns(prices):
    log_returns = np.log(prices[1:] / prices[:-1])
    return log_returns

def estimate_pdf(log_returns):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
    kde.fit(log_returns[:, np.newaxis])
    return kde

def compute_partition_function(kde, kF, TF, log_returns):
    ZF = np.sum(np.exp(-log_returns / (kF * TF)) * np.exp(kde.score_samples(log_returns[:, np.newaxis])))
    return ZF

def boltzmann_probability(log_returns, kde, ZF, kF, TF):
    probabilities = (np.exp(kde.score_samples(log_returns[:, np.newaxis])) * np.exp(-log_returns / (kF * TF))) / ZF
    return probabilities

def likelihood_function(params, log_returns, kde):
    kF, TF = params
    ZF = compute_partition_function(kde, kF, TF, log_returns)
    probabilities = boltzmann_probability(log_returns, kde, ZF, kF, TF)
    likelihood = -np.sum(np.log(probabilities))
    return likelihood

def optimize_parameters(log_returns, kde):
    result = differential_evolution(likelihood_function, bounds=[(-10, 10), (0.01, 5)], args=(log_returns, kde))
    kF_optimized, TF_optimized = result.x
    return kF_optimized, TF_optimized

def monte_carlo(prices, kF, TF, iterations, time_horizon):
    log_returns = calculate_log_returns(prices)
    kde = estimate_pdf(log_returns)
    ZF = compute_partition_function(kde, kF, TF, log_returns)
    future_prices = []
    
    for _ in range(iterations):
        path = [prices[-1]]
        for _ in range(time_horizon):
            sampled_return = np.random.choice(log_returns, p=boltzmann_probability(log_returns, kde, ZF, kF, TF) / np.sum(boltzmann_probability(log_returns, kde, ZF, kF, TF)))
            new_price = path[-1] * np.exp(sampled_return)
            path.append(new_price)
        future_prices.append(path)
    
    return np.array(future_prices)

prices = np.array([100, 102, 101, 103, 104, 105, 106, 108, 107, 109])
# in original version used available datasets

log_returns = calculate_log_returns(prices)

kde = estimate_pdf(log_returns)

kF_optimized, TF_optimized = optimize_parameters(log_returns, kde)
print(f"Optimized kF: {kF_optimized}, Optimized TF: {TF_optimized}")

iterations = 1000
time_horizon = 250
future_prices = monte_carlo(prices, kF_optimized, TF_optimized, iterations, time_horizon)

plt.figure(figsize=(10, 6))
plt.plot(future_prices.T, color='gray', alpha=0.1)
plt.plot(future_prices.mean(axis=0), color='red', label="Average Forecast")
plt.title("Monte Carlo Simulation - Asset Price Forecast")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()
