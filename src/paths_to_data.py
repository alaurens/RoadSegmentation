import os

"""
The differents paths to access our data allowing us to make sure there are no
issues when loading images
"""
FILE_PATH = os.path.dirname(__file__)
GROUNDTRUTH_PATH = FILE_PATH + '/../data/groundtruth/'
TRAIN_IMAGES_PATH = FILE_PATH + "/../data/images/"
TEST_FOLDER_PATH = FILE_PATH + '/../data/test_set_images/'
VALIDATION_IMAGES_PATH = FILE_PATH + "/../data/validation/"
PREDICTED_IMAGES_PATH = FILE_PATH + "/../data/prediction/"
LOGS_PATH = FILE_PATH + "/../logs/"
SUBMISSION_PATH = FILE_PATH + "/../submission/"
WEIGHTS_PATH = FILE_PATH + "/../weights/"
