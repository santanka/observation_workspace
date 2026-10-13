import unittest
import numpy as np
from core import interpolate, rotation, apply, series, FLAGS


def identity_transform(**kwargs):
    return kwargs['data_in']


class Tests(unittest.TestCase):
    def test_known_dsi_basis_and_component_order(self):
        # J2000=GSE=GSM in this synthetic case; spin +z, Sun +x -> R=identity.
        times=np.array([0., 8., 16.])
        r,d,_=rotation([4.,8.],times,np.array([[0.,90.]]*3),times,
                       np.array([[6371.2,0.,0.]]*3),transform=identity_transform)
        np.testing.assert_allclose(r,np.broadcast_to(np.eye(3),r.shape),atol=1.e-15)
        v=np.array([[200.,-30.,10.],[-20.,40.,-3.]])
        out,valid=apply(r,v)
        np.testing.assert_allclose(out,v,atol=1.e-12)
        self.assertTrue(valid.all())

    def test_nontrivial_rotation_and_inverse(self):
        # spin axis +y, Sun +x: DSI x=+x, y=-z, z=+y.
        times=np.array([0.,8.])
        r,d,_=rotation([4.],times,np.array([[90.,0.]]*2),times,
                       np.array([[6371.2,0.,0.]]*2),transform=identity_transform)
        v=np.array([[1.,2.,3.]])
        out,_=apply(r,v)
        np.testing.assert_allclose(out,[[1.,3.,-2.]],atol=1.e-12)
        np.testing.assert_allclose(np.einsum('nji,nj->ni',r,out),v,atol=1.e-12)

    def test_no_extrapolation_and_no_cross_gap_interpolation(self):
        y=np.array([[1.,0.,0.],[1.,0.,0.],[1.,0.,0.]])
        out,valid,b=interpolate([0.,8.,100.],y,[-1.,0.,4.,8.,20.,100.,101.],16.)
        np.testing.assert_array_equal(valid,[False,True,True,True,False,True,False])
        self.assertTrue(np.isnan(out[~valid]).all())
        self.assertEqual(b['gap_seconds'][4],92.)

    def test_nan_endpoint_not_bridged_and_exact_record_allowed(self):
        out,valid,_=interpolate([0.,8.,16.],[[1.,0.,0.],[np.nan,0.,0.],[1.,0.,0.]],
                                [0.,4.,8.,12.,16.],16.)
        np.testing.assert_array_equal(valid,[True,False,False,False,True])

    def test_degenerate_sun_parallel_spin_and_flags(self):
        times=[0.,8.]
        r,d,_=rotation([4.],times,[[0.,0.]]*2,times,[[6371.2,0.,0.]]*2,
                       transform=identity_transform)
        self.assertEqual(d['rotation_quality'][0],FLAGS['degenerate_dsi_basis'])
        self.assertTrue(np.isnan(r).all())

    def test_conflicting_duplicates_and_reversed_times_rejected(self):
        with self.assertRaises(ValueError): series([0.,0.],[[1.],[2.]])
        with self.assertRaises(ValueError): series([1.,0.],[[1.],[1.]])
        t,v=series([0.,0.,8.],np.array(['I','I','V']))
        np.testing.assert_array_equal(v,['I','V'])

    def test_velocity_nan_and_orbit_fill_not_hidden(self):
        times=[0.,8.]
        r,d,_=rotation([0.,4.],times,[[0.,90.]]*2,times,[[6371.2,0.,0.],[-1e31,0.,0.]],
                       transform=identity_transform)
        self.assertEqual(d['rotation_quality'][1],FLAGS['orbit_unavailable'])
        out,valid=apply(r,[[np.nan,0.,0.],[200.,0.,0.]])
        self.assertFalse(valid.any()); self.assertTrue(np.isnan(out).all())


if __name__=='__main__': unittest.main()
