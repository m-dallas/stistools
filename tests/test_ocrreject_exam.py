from stistools.ocrreject_exam import ocrreject_exam
from .resources import BaseSTIS
import numpy as np
import os
import pytest


@pytest.mark.bigdata
@pytest.mark.slow
class TestOcrrejectExam(BaseSTIS):

    input_loc = 'ocrreject_exam'

    input_list = ["odvkl1040_flt.fits", "odvkl1040_sx1.fits"]

    # Make input file string
    input_file_string = ", ".join(input_list)

    def test_ocrrject_exam(self):
        """
        This regression test runs the task on the known problematic dataset odvkl1040.
        The resulting output dictionary is compared to the known values for odvkl1040.
        """

        # Prepare input files.
        for filename in self.input_list:
            local_file = self.get_data("input", filename)
        
        expected_output = {'rootname': 'odvkl1040',
                            'n_splits': 2,
                            'detector_box_fraction' : 0.0078125,
                            'extr_fracs': np.array([0.31530762, 0.32006836]),
                            'outside_fracs': np.array([0.00884673, 0.00810278]), 
                            'ratios': np.array([35.64113429, 39.50106762]),
                            'n_cr_pix':np.array([11787, 11052]),
                            'avg_extr_frac': 0.31768798828125, 
                            'avg_outside_frac': 0.008474755474901575,
                            'avg_ratio': 37.486389928547126,
                            'max_ratio': 39.501067615658364,
                            'max_ratio_ncr_pix': 11052,
                            'overflagged_max_ratio': 1.30018281535649,
                            'likely_overflagged': True}

        resulting_output = ocrreject_exam('odvkl1040', data_dir=os.path.dirname(local_file))

        assert len(resulting_output) == 1, "Output is not a list of only one dictionary"

        resulting_output = resulting_output[0]

        assert set(resulting_output) == set(expected_output), "Not all created keys match"

        for key in expected_output:
            assert resulting_output[key] == pytest.approx(expected_output[key]), f"{key} failed"
