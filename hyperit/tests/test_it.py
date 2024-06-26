import unittest
from unittest.mock import MagicMock, PropertyMock, patch
import numpy as np
from hyperit.hyperit import HyperIT 
from hyperit.utils import convert_names_to_indices, convert_indices_to_names
import os

class TestHyperIT(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        HyperIT.setup_JVM(verbose=True)

    def setUp(self):
        """Set up test variables used in the tests."""
        self.channels = [['C1', 'C2', 'C3'], ['C1', 'C2', 'C3']]
        self.data1 = np.random.rand(10, 3, 600)  # 10 epochs, 3 channels, 100 samples
        self.data2 = np.random.rand(10, 3, 600)
        self.hyperit_instance = HyperIT(self.data1, self.data2, self.channels)

    def test_check_data_valid(self):
        """Test the data validation logic with correct input."""
        try:
            HyperIT(self.data1, self.data2, self.channels)
        except Exception as e:
            self.fail(f"Initialization failed with correct data: {e}")

    def test_check_data_invalid_shape(self):
        """Test the data validation logic with incorrect input shapes."""
        data_wrong = np.random.rand(2,2,2,100)
        with self.assertRaises(ValueError):
            HyperIT(data_wrong, data_wrong, self.channels)

    @patch.object(HyperIT, 'roi', new_callable=PropertyMock)
    def test_roi_setting(self, mock_roi):
        """Test setting ROI correctly assigns indices."""
        mock_roi.return_value = [[0, 1], [1, 2]]
        self.assertEqual(self.hyperit_instance.roi, [[0, 1], [1, 2]])

    def test_reset_roi(self):
        """Test resetting ROI to all channels."""
        self.hyperit_instance.roi = [['C1', 'C2'], ['C2', 'C3']]
        self.hyperit_instance.reset_roi()
        expected_roi = []
        self.assertTrue(np.array_equal(self.hyperit_instance.roi, expected_roi))

    def test_mi_computation(self):
        """Test Mutual Information computation."""
        newitmi = HyperIT(self.data1, self.data2, self.channels)
        newitmi.compute_mi('histogram', include_intra=True)
        self.assertIsNotNone(newitmi._it_matrix)

    def test_te_computation(self):
        """Test Transfer Entropy computation setup."""
        newitte = HyperIT(self.data1, self.data2, self.channels)
        newitte.compute_te('gaussian', include_intra=False)
        self.assertIsNotNone(newitte._it_matrix)

    def test_compute_atoms_execution(self):
        """Test that compute_atoms executes and returns data."""
        try:
            matrix = self.hyperit_instance.compute_atoms()
            self.assertIsNotNone(matrix, "matrix should not be None")
            print("compute_atoms method executed successfully.")
        except Exception as e:
            self.fail(f"compute_atoms method failed with an exception {e}")

    
    @patch('matplotlib.pyplot.show')  
    def test_plotting(self, mock_plot_show):
        """Test the plotting function calls."""
        self.hyperit_instance.compute_mi('histogram', vis=True)
        mock_plot_show.assert_called() 

    def tearDown(self):
        """Clean up any mock patches to prevent leaks between tests."""
        patch.stopall()



class TestConvertNamesToIndices(unittest.TestCase):

    def setUp(self):
        self.channel_names = [['C1', 'C2', 'C3', 'C4'], ['C1', 'C2', 'C3', 'C4']]

    def test_grouped_comparison(self):
        roi = [['C1', 'C3'], ['C2', 'C4']]
        expected = [[0, 2], [1, 3]]
        result = convert_names_to_indices(self.channel_names, roi, 1)
        self.assertEqual(result, expected)

    def test_pointwise_comparison(self):
        roi = ['C2', 'C3']
        expected = [1, 2]
        result = convert_names_to_indices(self.channel_names, roi, 0)
        self.assertEqual(result, expected)

    def test_single_channel(self):
        roi = 'C3'
        expected = [2]
        result = convert_names_to_indices(self.channel_names, roi, 0)
        self.assertEqual(result, expected)

    def test_direct_index_input(self):
        roi = [1, 2]
        expected = [1, 2]
        result = convert_names_to_indices(self.channel_names, roi, 0)
        self.assertEqual(result, expected)

    def test_invalid_channel_name(self):
        roi = ['C5']  # Does not exist in participant 0's list
        with self.assertRaises(ValueError):
            convert_names_to_indices(self.channel_names, roi, 0)

class TestConvertIndicesToNames(unittest.TestCase):

    def setUp(self):
        # Setting up channel names for testing, identical for both participants for simplicity.
        self.channel_names = [['C1', 'C2', 'C3', 'C4'], ['C1', 'C2', 'C3', 'C4']]

    def test_grouped_comparison(self):
        # Testing conversion from indices back to names for grouped indices.
        indices = [[0, 2], [1, 3]]
        expected = [['C1', 'C3'], ['C2', 'C4']]
        result = convert_indices_to_names(self.channel_names, indices, 1)
        self.assertEqual(result, expected)

    def test_pointwise_comparison(self):
        # Testing conversion for pointwise indices.
        indices = [1, 2]
        expected = ['C2', 'C3']
        result = convert_indices_to_names(self.channel_names, indices, 0)
        self.assertEqual(result, expected)

    def test_single_channel(self):
        # Testing conversion when a single channel index is given.
        indices = [2]
        expected = ['C3']
        result = convert_indices_to_names(self.channel_names, indices, 0)
        self.assertEqual(result, expected)

    def test_direct_name_input(self):
        # Testing the scenario where channel names are directly given instead of indices.
        # Assuming your function should handle or convert directly if names are passed (Optional based on your function definition).
        names = ['C2', 'C3']
        expected = ['C2', 'C3']
        result = convert_indices_to_names(self.channel_names, names, 0)
        self.assertEqual(result, expected)

    def test_invalid_index(self):
        # Testing behavior with an index that is out of range (not present in channel_names list).
        indices = [4]  # Out of range index as lists only go up to index 3 (C4)
        with self.assertRaises(IndexError):
            convert_indices_to_names(self.channel_names, indices, 0)


if __name__ == '__main__':
    unittest.main()
