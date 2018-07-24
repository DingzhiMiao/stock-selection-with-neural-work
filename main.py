import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from pre_utils import create_dataset, load_dataset

i_month = 83
create_dataset(i_month)
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, train_set_stock, test_set_stock = load_dataset(i_month)

