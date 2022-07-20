import pytest
from packaging.version import Version

from dipy.data import get_fnames
from dipy.nn.synb0 import Synb0
from dipy.utils.optpkg import optional_package
import numpy as np
from numpy.testing import assert_almost_equal

tf, have_tf, _ = optional_package('tensorflow')
tfa, have_tfa, _ = optional_package('tensorflow_addons')

if have_tf and have_tfa:
    if Version(tf.__version__) < Version('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
@pytest.mark.skipif(not have_tfa, reason='Requires TensorFlow_addons')
def test_default_weights():
    file_names = get_fnames('synb0_test_data')
    input_arr1 = np.load(file_names[0])[0, :, :, :, 0]
    input_arr2 = np.load(file_names[0])[0, :, :, :, 1]
    target_arr = np.load(file_names[1])[0]

    synb0_model = Synb0()

    results_arr = synb0_model.predict(input_arr1, input_arr2, average=True)
    assert_almost_equal(results_arr, target_arr)

def test_default_weights_batch():
    file_names = get_fnames('synb0_test_data')
    input_arr1 = np.load(file_names[0])[..., 0]
    input_arr2 = np.load(file_names[0])[..., 1]
    target_arr = np.load(file_names[1])

    synb0_model = Synb0()
    results_arr = synb0_model.predict(input_arr1, input_arr2, batch_size=2, average=True)
    assert_almost_equal(results_arr, target_arr)