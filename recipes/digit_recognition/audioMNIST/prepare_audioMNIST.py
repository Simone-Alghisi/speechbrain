"""
Downloads and creates data manifest files for audioMNIST (digit recognition).
Iterations of digits from 0 to 4 are put into the test set, while instead
train and validation data are prepared by splitting iterations from 5 to 50
using scikit-learn train_test_split. Overall, three sets are created.

Authors:
 * Simone Alghisi, 2022
"""

import os
import json
import shutil
import random
import logging
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
AUDIOMNIST_DATASET_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/tags/v1.0.10.tar.gz"
SAMPLERATE = 16000


def prepare_audio_mnist(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    train_ratio=0.9,
):
    """
    Prepares the json files for the audio MNIST dataset.

    Downloads the dataset if it is not found in the `data_folder`.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the audio MNIST dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    split_ratio: float
        A single float indicating the ratio of data between original and validation:
        the train ratio will be equal to train_ratio, while the valid ratio will be
        equal to 1 - train_ratio.

    Example
    -------
    >>> data_folder = '/path/to/audio_MNIST'
    >>> prepare_audio_mnist(data_folder, 'train.json', 'valid.json', 'test.json')
    """

    # Check if this phase is already done (if so, skip it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    # If the dataset doesn't exist yet, download it
    train_folder = os.path.join(data_folder, "audioMNIST")
    if not check_folders(train_folder):
        download_audio_mnist(data_folder)
    else:
        print(f"{train_folder} exists, skipping.")

    # List files and create manifest from list
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )
    extension = [".wav"]
    wav_list = get_all_files(train_folder, match_and=extension)

    # Create splits based on MNIST guidelines and train_ratio
    data_split = split_mnist_data(wav_list, train_ratio=train_ratio)

    # Creating json files
    create_json(data_split["train"], save_json_train)
    create_json(data_split["valid"], save_json_valid)
    create_json(data_split["test"], save_json_test)


def create_json(wav_list, json_file):
    """
    Creates the json file given a list of wav files.

    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    """
    # Processing all the wav files in the list
    json_dict = {}
    for wav_file in wav_list:

        # Reading the signal (to retrieve duration in seconds)
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # Manipulate path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-2:])

        # Getting digit-id from utterance-id
        digit_id = uttid.split("_")[0]

        # Create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "digit_id": digit_id,
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def split_mnist_data(wav_list, train_ratio):
    """Splits the wav list into training, and test lists as suggested by MNIST.
    Then, it performs an additional random split on the training set to obtain the
    validation list.

    Arguments
    ---------
    wav_lsit : list
        list of all the signals in the dataset
    split_ratio: float
        A single float indicating the ratio of data between original and validation:
        the train ratio will be equal to train_ratio, while the valid ratio will be
        equal to 1 - train_ratio.

    Returns
    ------
    dictionary containing train, valid, and test splits.
    """

    data_splits = {"train": [], "valid": [], "test": []}

    for wav in wav_list:
        # retrieve the iteration number
        iter_number = int(wav.split("_")[-1].split(".")[0])
        # split accordingly to MNIST
        if iter_number < 5:
            data_splits["test"].append(wav)
        else:
            data_splits["train"].append(wav)

    random.shuffle(data_splits["train"])
    tot_snts = len(data_splits["train"])

    n_snts = int(tot_snts * (1 - train_ratio))
    data_splits["valid"] = data_splits["train"][0:n_snts]
    del data_splits["train"][0:n_snts]

    return data_splits


def download_audio_mnist(destination):
    """Download dataset repo, unpack it, and remove unnecessary elements.

    Arguments
    ---------
    destination : str
        Place to put dataset.
    """
    audio_mnist_archive = os.path.join(destination, "audioMNISTRepo.tar.gz")
    download_file(AUDIOMNIST_DATASET_URL, audio_mnist_archive)
    shutil.unpack_archive(audio_mnist_archive, destination)
    audio_mnist_repo = os.path.join(
        destination, "free-spoken-digit-dataset-1.0.10"
    )
    audio_mnist = os.path.join(destination, "audioMNIST")

    shutil.move(os.path.join(audio_mnist_repo, "recordings"), audio_mnist)
    shutil.rmtree(audio_mnist_repo)


if __name__ == "__main__":
    prepare_audio_mnist("data", "train.json", "valid.json", "test.json")
