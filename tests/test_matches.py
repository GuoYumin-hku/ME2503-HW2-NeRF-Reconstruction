import unittest
import numpy as np

from hw2.match_features import calibrated_geometry, sampson_errors


class CalibratedGeometryTests(unittest.TestCase):
    def test_known_cameras_predict_projected_correspondences(self):
        k = np.array([[100., 0., 64.], [0., 100., 64.], [0., 0., 1.]])
        first = np.eye(4)
        second = np.eye(4)
        second[0, 3] = 1.
        points = np.array([[0., .1, -4.], [.3, -.2, -5.], [-.4, .3, -3.],
                           [.1, -.4, -4.5], [.5, .4, -3.5], [-.2, -.1, -4.2],
                           [.15, .25, -6.], [-.3, .2, -5.5]])

        def project(c2w):
            p = (points - c2w[:3, 3]) @ c2w[:3, :3]
            return np.column_stack([64. + 100. * p[:, 0] / -p[:, 2],
                                    64. - 100. * p[:, 1] / -p[:, 2]])

        x1, x2 = project(first), project(second)
        f, rotation, translation = calibrated_geometry(
            {'transform_matrix': first.tolist()}, {'transform_matrix': second.tolist()}, k, k)
        self.assertLess(float(sampson_errors(f, x1, x2).max()), 1e-9)
        self.assertTrue(np.allclose(rotation, np.eye(3)))
        self.assertTrue(np.allclose(translation, [-1., 0., 0.]))


if __name__ == '__main__': unittest.main()
