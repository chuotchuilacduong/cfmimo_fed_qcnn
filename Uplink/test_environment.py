import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Environment.Environment import Environment
def test_large_scale_fading():
    env = Environment(M=3,K=5)
    large_scale_fading = env.compute_large_scale_fading()
    
    assert large_scale_fading.shape == (env.M, env.K), "Matrix shape is incorrect"
    
    # Check for any NaN or infinite values
    assert not np.isnan(large_scale_fading).any(), "Matrix contains NaN values"
    assert not np.isinf(large_scale_fading).any(), "Matrix contains infinite values"

if __name__ == "__main__":
    test_large_scale_fading()
