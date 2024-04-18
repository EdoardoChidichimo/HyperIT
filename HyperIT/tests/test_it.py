import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from hyperit import HyperIT 
import os

class TestHyperIT(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.jarLocation = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'infodynamics.jar')
        HyperIT.setup_JVM(cls.jarLocation, verbose=True)

    def setUp(self):
        """Set up test variables used in the tests."""
        self.channels = [['C1', 'C2', 'C3'], ['C1', 'C2', 'C3']]
        self.data1 = np.random.rand(10, 3, 600)  # 10 epochs, 3 channels, 100 samples
        self.data2 = np.random.rand(10, 3, 600)
        self.freq_bands = {'alpha': (8, 12)}
        self.sfreq = 256  # Hz

    @patch('hyperit.HyperIT.setup_JVM')
    def test_initialization(self, mock_setup_jvm):
        """Test object initialization and JVM setup call."""
        hyperit_instance = HyperIT(data1=self.data1, 
                                   data2=self.data2, 
                                   channel_names=self.channels, 
                                   sfreq=self.sfreq, 
                                   freq_bands=self.freq_bands)
        mock_setup_jvm.assert_not_called()
        self.assertEqual(hyperit_instance._sfreq, self.sfreq)

    def test_check_data_valid(self):
        """Test the data validation logic with correct input."""
        try:
            HyperIT(self.data1, self.data2, self.channels, self.sfreq, self.freq_bands)
        except Exception as e:
            self.fail(f"Initialization failed with correct data: {e}")

    def test_check_data_invalid_shape(self):
        """Test the data validation logic with incorrect input shapes."""
        data_wrong = np.random.rand(5, 100)  # Wrong shape
        with self.assertRaises(ValueError):
            HyperIT(data_wrong, data_wrong, self.channels, self.sfreq, self.freq_bands, working_directory=self.jarLocation)

    @patch('hyperit.convert_names_to_indices', return_value=[0, 1, 2])
    def test_roi_setting(self, mock_convert):
        """Test setting ROI correctly assigns indices."""
        hyperit_instance = HyperIT(self.data1, self.data2, self.channels, self.sfreq, self.freq_bands, working_directory=self.jarLocation)
        hyperit_instance.roi = [[['C1', 'C2']], [['C2', 'C3']]]
        self.assertEqual(hyperit_instance._roi, [[0, 1], [1, 2]])

    @patch('hyperit.convert_names_to_indices', return_value=[0, 1, 2])
    def test_reset_roi(self, mock_convert):
        """Test resetting ROI to all channels."""
        hyperit_instance = HyperIT(self.data1, self.data2, self.channels, self.sfreq, self.freq_bands, working_directory=self.jarLocation)
        hyperit_instance.roi = [[['C1', 'C2']], [['C2', 'C3']]]
        hyperit_instance.reset_roi()
        expected_roi = [np.arange(3), np.arange(3)]
        self.assertTrue(np.array_equal(hyperit_instance._roi[0], expected_roi[0]) and
                        np.array_equal(hyperit_instance._roi[1], expected_roi[1]))

    @patch('hyperit.np.histogram2d', return_value=(np.zeros((10, 10)), None, None))
    @patch('hyperit.stats.iqr', return_value=1.0)
    def test_mi_computation(self, mock_hist, mock_iqr):
        """Test Mutual Information computation."""
        hyperit_instance = HyperIT(self.data1, self.data2, self.channels, self.sfreq, self.freq_bands, verbose=True, working_directory=self.jarLocation)
        hyperit_instance.compute_mi('histogram')
        self.assertIsNotNone(hyperit_instance.mi_matrix)
        self.assertTrue(mock_hist.called)
        self.assertTrue(mock_iqr.called)

    @patch('hyperit.setup_JArray', return_value=None)
    @patch('hyperit.set_estimator', return_value=('kernel', MagicMock(), {'prop1': 'value1'}, (2,)))
    def test_te_computation(self, mock_set_estimator, mock_jarray):
        """Test Transfer Entropy computation setup."""
        hyperit_instance = HyperIT(self.data1, self.data2, self.channels, self.sfreq, self.freq_bands, verbose=True, working_directory=self.jarLocation)
        te_xy, te_yx = hyperit_instance.compute_te('kernel')
        self.assertIsNotNone(te_xy)
        self.assertIsNotNone(te_yx)
        self.assertTrue(mock_set_estimator.called)
        self.assertEqual(mock_set_estimator.call_args[0], ('kernel', 'te', {}))

    @patch('hyperit.calc_PhiID', return_value=({}, None))
    def test_phiid_computation(self, mock_phiid):
        """Test Integrated Information Decomposition computation."""
        hyperit_instance = HyperIT(self.data1, self.data2, self.channels, self.sfreq, self.freq_bands, verbose=True, working_directory=self.jarLocation)
        phi_xy, phi_yx = hyperit_instance.compute_atoms()
        self.assertIsNotNone(phi_xy)
        self.assertIsNotNone(phi_yx)
        self.assertTrue(mock_phiid.called)

    @patch('builtins.input', return_value='1')  # Simulates choosing "1. All epochs"
    def test_plotting(self, mock_plot_show):
        """Test the plotting function calls."""
        hyperit_instance = HyperIT(self.data1, self.data2, self.channels, self.sfreq, self.freq_bands, verbose=True, working_directory=self.jarLocation)
        hyperit_instance.compute_mi('histogram', vis=True)
        mock_plot_show.assert_called()

    def tearDown(self):
        """Clean up any mock patches to prevent leaks between tests."""
        patch.stopall()

if __name__ == '__main__':
    unittest.main()
