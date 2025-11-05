import pytest
import numpy as np
from pathlib import Path

from pyflowmeld.utils.benchmarks import OverlappingSpherePack

@pytest.fixture
def temp_dir(tmp_path):
    yield tmp_path

class TestOverlappingSpherePack:
    """
    Tests for OverlappingSpherePack benchmark class
    """

    def test_initialization_basic(self, temp_dir):
        shape = (16, 12, 7)
        num_spheres = 10
        porosity = 0.4
        bench = OverlappingSpherePack(
            domain_size=shape, 
            num_spheres=num_spheres,
            porosity=porosity,
            delta=1.0,
            save=True,
            save_path=temp_dir
        )
        assert bench.domain.shape == shape
        assert bench.domain_volume == np.prod(shape)
        assert bench.seed_coordinates.shape == (num_spheres, 3)
        assert bench.porosity == porosity
        assert isinstance(bench.save_path, Path)
        assert np.array_equal(bench.drainage_side_walls, np.zeros(6)) or bench.drainage_side_walls == [0]*6

    @pytest.mark.parametrize("bad_num_spheres", [0, -3, "notint"])
    def test_invalid_num_spheres_raises(self, bad_num_spheres):
        shape = (12, 12, 12)
        with pytest.raises(ValueError):
            OverlappingSpherePack(domain_size=shape, num_spheres=bad_num_spheres)

    @pytest.mark.parametrize("bad_poly", [0.9, "str", -5])
    def test_invalid_poly_max_factor_raises(self, bad_poly):
        shape = (12, 10, 8)
        with pytest.raises(ValueError):
            OverlappingSpherePack(domain_size=shape, num_spheres=5, poly_max_factor=bad_poly)

    @pytest.mark.parametrize("bad_axes", [
        (0,), ("x", 1), (5, 7), (0, 1, 2)
    ])
    def test_invalid_drainage_swap_axes_raises(self, bad_axes):
        shape = (8, 8, 8)
        with pytest.raises(ValueError):
            OverlappingSpherePack(domain_size=shape, num_spheres=5, drainage_swap_axes=bad_axes)

    @pytest.mark.parametrize("bad_side_walls", [
        [-1]*6, "string", (1,2,3,4), [1,2,3,4,5,6,7]
    ])
    def test_invalid_drainage_side_walls_raises(self, bad_side_walls):
        shape = (10, 10, 10)
        with pytest.raises(ValueError):
            OverlappingSpherePack(domain_size=shape, num_spheres=4, drainage_side_walls=bad_side_walls)