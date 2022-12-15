# Recipy for Digit Recognition

> Note:
> - This work has been developed as an assignment to work on the SpeechBrain python library.
> - The recipy has been developed starting from the speaker_id template. [For more information, please take a look into the "speaker-id from scratch" tutorial](https://colab.research.google.com/drive/1UwisnAjr8nQF3UnrkIJ4abBMAWzVwBMh?usp=sharing)

This folder provides a working, well-documented example for training
a digit recognition model from scratch data. The data we use is from
[Audio MNIST free spoken digit dataset](https://github.com/Jakobovski/free-spoken-digit-dataset) + OpenRIR.

There are four files here:

* `train.py`: the main code file, outlines the entire training process.
* `train.yaml`: the hyperparameters file, sets all parameters of execution.
* `mini_librispeech_prepare.py`: If necessary, downloads and prepares data manifests.

Instead, the model is available in `speechbrain/lobes/models`:
* `resnet.py`: A file containing a modified version of TorchVision ResNet.

To train the digit recognition model, just execute the following on the command-line:

```bash
python train.py train.yaml
```

This will automatically download and prepare the data manifest for Audio 
MNIST, and then train a model with dynamically augmented samples.

More details about what each file does and how to make modifications
are found within each file. 
