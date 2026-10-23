"""Input contracts for external examples; no QEDispInv checkout is required."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from check_qedisp_examples import model_table, data_table, check_queries


class ExternalContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'input.txt'

    def test_model_units_order_and_duplicate_display_index(self):
        self.path.write_text('7 0 1.8 .4 .8\n7 .01 2.1 .2 .5\n')
        model = model_table(self.path)
        self.assertEqual(model, dict(depth=[0., .01], vs=[.4, .2], vp=[.8, .5], rho=[1.8, 2.1]))
        # Low velocity layers are allowed; crossing interfaces are not.
        self.path.write_text('0 0 1.8 .4 .8\n1 0 2.1 .2 .5\n')
        with self.assertRaises(ValueError):
            model_table(self.path)

    def test_errors_are_explicit_and_input_order_is_preserved(self):
        original = np.array([[12., .3, 2], [6., .4, 0], [12., .3, 2]])
        np.savetxt(self.path, original)
        for sigma in [None, 0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                data_table(self.path, sigma)
        converted = data_table(self.path, .005)
        np.testing.assert_array_equal(converted[:, :3], original)
        np.testing.assert_array_equal(converted[:, 3], [.005]*3)
        converted[:, 3] = [.001, .002, .003]
        np.savetxt(self.path, converted)
        np.testing.assert_array_equal(data_table(self.path, .1), converted)

    def test_fractional_mode_rejected(self):
        self.path.write_text('12 .3 1.5\n')
        with self.assertRaises(ValueError):
            data_table(self.path, .005)

    def test_no_dropped_or_reordered_forward_rows(self):
        rows = [['0', '6', '1'], ['1', '12', '0']]
        check_queries(rows, [6., 12.], [1, 0])
        for bad in [rows[:1], rows[::-1], [['0', '6', '0'], rows[1]]]:
            with self.assertRaises(ValueError):
                check_queries(bad, [6., 12.], [1, 0])


if __name__ == '__main__':
    unittest.main()
