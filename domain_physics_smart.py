import domain_physics
from domain_physics import size_d

size_d = size_d

def make_env_app_tree(**kwargs):
    return domain_physics.make_env_app_tree(smart_physics=True, **kwargs)
