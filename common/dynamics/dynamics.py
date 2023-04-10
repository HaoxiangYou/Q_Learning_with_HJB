import numpy as np
from typing import Tuple
class Dynamics:
    def __init__(self,) -> None:
        pass
    
    def get_dimension(self,) -> Tuple[int, int]:
        raise NotImplementedError

    def get_control_affine_matrix(self, x) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    def get_control_limit(self,) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError 