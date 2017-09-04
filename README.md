# RenderGAN

This repository holds the code to the ["RenderGAN: Generating Realistic Labeled Data"](https://arxiv.org/abs/1611.01331)
paper.

You might be interested in:

* implementation of the [RenderGAN](https://github.com/berleon/deepdecoder/blob/master/deepdecoder/render_gan.py#L130)
* code of the [generator network](https://github.com/berleon/deepdecoder/blob/master/deepdecoder/render_gan.py#L168)
* [discriminator](https://github.com/berleon/deepdecoder/blob/master/deepdecoder/networks.py#L480)
* architecture of the [decoder network](https://github.com/berleon/deepdecoder/blob/master/deepdecoder/networks.py#L588)
* the [Makefile](https://github.com/berleon/deepdecoder/blob/master/deepdecoder/scripts/Makefile) to orchestrate the training process
* the [hyperparameters](https://github.com/berleon/deepdecoder/tree/master/config/train) for the training

## Run the code

1. Build the docker image as described in [docker/README.md](docker/README.md)
1. Start a docker container and ensure you have enough disk space ~500GB.
1. Mount or clone this repository into the docker container.
1. Install the repository with `pip3 install -e . `.
1. Download all files from this [google drive directory](https://drive.google.com/drive/folders/0B4-Jw9T9VL8nakJwa1U0YU9nUVU) and place them in the `$REPO_DIR/data`
   directory. You might find the [gdrive cli tool](https://github.com/prasmussen/gdrive) useful.
   Check the sha1 sums:
    ```
    db00d1ae8d1920c6d4faa830cebcd5783b44e905  gt_test_shuffled.hdf5
    37961551dd062179cafc3f7eee3a75e113555c59  gt_train_shuffled.hdf5
    be1870252dd926aeff8818ad13d9b28a1253935e  real_tags.hdf5
    ```

1. Now your can run the code.
    ```bash
    $ mkdir build
    $ cd build
    $ echo '{"path": "../../data/real_tags.hdf5"}' > real_dataset
    $ bb_make ../config/train tag3d_network
    ```
   The ``bb_make`` program is a s lim wrapper around this [Makefile](https://github.com/berleon/deepdecoder/blob/master/deepdecoder/scripts/Makefile).
   The first argument is the directory of the parameters you want to use and the
   second is the make target to execute.

   Targets:
    * **tag3d_data_set**: Generates a dataset using the simple 3D model.
    * **tag3d_network**: Train a neural network to emulate the simple 3D model.
    * **rendergan**: Trains the GAN.
    * **artificial_data_set**: Samples an artificial dataset from the RenderGAN.
    * **decoder_model**: Trains a neural network to decode the tags.
    * **decoder_evaluation**: Evaluates a trained decoder network and prints the mean hamming distance.
    * **decoder_on_real**: Train a neural network on the limited set of real data.

   Every target has a corresponding setting file (e.g. setting the number of sample to generate ...). The `decoder_model` target can be run with different settings using the `-o` flag:
    ```
    $ bb_make ../config/train -o decoder_settings=decoder_handmade_tag3d.yaml decoder
    ```
   This will execute the decoder target but with the `decoder_handmade_tag3d.yaml` file.

   You can also copy the settings directory `$ cp -r config/train
   config/my_settings` and experiement with your own hyperparameters. Then you
   have to use `$ bb_make ../config/my_settings`
