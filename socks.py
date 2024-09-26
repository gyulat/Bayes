import numpy as np
import pandas as pd
from scipy.stats import nbinom, beta

# Initialize parameters
n_picked = 11  # The number of socks we are going to pick

# Function for simulating sock picking
def simulate_sock_picking(n_picked):
    prior_mu = 30
    prior_sd = 15
    prior_size_param = -prior_mu**2 / (prior_mu - prior_sd**2)
    n_socks = nbinom.rvs(n=prior_size_param, p=prior_mu / (prior_mu + prior_size_param), size=1)[0]
    prop_pairs = beta.rvs(15, 2)
    n_pairs = round(np.floor(n_socks / 2) * prop_pairs)
    n_odd = n_socks - n_pairs * 2

    # Create socks array
    socks = np.repeat(np.arange(1, n_pairs + n_odd + 1), [2] * n_pairs + [1] * n_odd)
    np.random.shuffle(socks)

    # Pick socks
    picked_socks = np.random.choice(socks, size=min(n_picked, n_socks), replace=False)
    sock_counts = pd.Series(picked_socks).value_counts()

    # Return counts and parameters
    return {
        'unique': np.sum(sock_counts == 1),
        'pairs': np.sum(sock_counts == 2),
        'n_socks': n_socks,
        'n_pairs': n_pairs,
        'n_odd': n_odd,
        'prop_pairs': prop_pairs
    }

# Simulating the process 10000 times
simulations = [simulate_sock_picking(n_picked) for _ in range(10000)]

# Convert to DataFrame for easier handling
sock_sim_df = pd.DataFrame(simulations)

# Filter for specific case
post_samples_case1 = sock_sim_df[(sock_sim_df['unique'] == 11) & (sock_sim_df['pairs'] == 0)]

# Print the first few rows of the simulations
print(sock_sim_df.head())

# Median of the number of socks
print("number of socks: ", sock_sim_df['n_socks'].median())

# Histogram of the number of socks
sock_sim_df.hist(column='n_socks')
