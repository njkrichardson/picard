from os.path import dirname, abspath, join

__all__ = []

# Paths 
SRC = dirname(abspath(__file__)) 

# Caching configuration 
cache_config = dict(
        root=join(SRC, "cached"),
        top_level=["regression", "classification", "minimization"]
        ) 
