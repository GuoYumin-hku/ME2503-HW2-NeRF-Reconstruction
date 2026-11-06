import unittest
import numpy as np
import torch
from hw2.nerf import make_rays, volume_composite, generate_split, diverse_view_indices

class GeometryTests(unittest.TestCase):
    def test_camera_axis_and_image_orientation(self):
        c2w = torch.eye(4)
        origins, dirs = make_rays(3, 3, 2.0, c2w)
        self.assertTrue(torch.allclose(origins[1, 1], torch.zeros(3)))
        self.assertTrue(torch.allclose(dirs[1, 1], torch.tensor([0., 0., -1.])))
        self.assertGreater(dirs[0, 0, 1], 0)
        self.assertLess(dirs[0, 0, 0], 0)

    def test_composite_transparent_and_opaque(self):
        rgb = torch.tensor([[[1., 0., 0.], [0., 0., 1.]]])
        delta = torch.ones(1, 2)
        transparent, _, _ = volume_composite(rgb, torch.zeros(1, 2), delta)
        opaque, _, _ = volume_composite(rgb, torch.tensor([[100., 0.]]), delta)
        self.assertTrue(torch.allclose(transparent, torch.ones(1,3), atol=1e-5))
        self.assertTrue(torch.allclose(opaque, torch.tensor([[1.,0.,0.]]), atol=1e-5))

    def test_held_out_never_enters_train_split(self):
        frames = [{'split':'train','i':1}, {'split':'test','i':2}, {'split':'train','i':3}]
        self.assertEqual([x['i'] for x in generate_split(frames,'train')], [1,3])
        self.assertEqual([x['i'] for x in generate_split(frames,'test')], [2])

    def test_sparse_views_cover_opposite_directions(self):
        positions = np.array([[1.,0.,0.],[.9,.1,0.],[-1.,0.,0.],[0.,1.,0.]])
        self.assertEqual(diverse_view_indices(positions, 2), [0, 2])

if __name__ == '__main__': unittest.main()
