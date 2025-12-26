"""
DEPRECATED: The logic from this file has been moved to scripts/compile_dataset.py
as part of the sequential filtering and hard negative mining refactor.
"""

def load_audit_results(input_file):
    raise DeprecationWarning("This function is deprecated. Load raw data directly.")

def create_pools(df):
    raise DeprecationWarning("This function is deprecated.")

def compile_risk_set(train_risk, min_purity=0.1):
    raise DeprecationWarning("This function is deprecated.")

def compile_safe_set(train_safe, max_risk_density=0.9):
    raise DeprecationWarning("This function is deprecated.")

def create_ambiguous_test_set(test_risk, test_safe, size=1000):
    raise DeprecationWarning("This function is deprecated.")
