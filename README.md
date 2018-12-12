# Getting Started with PyTorch

A simple repo to help beginers get started with Pytorch, there is also plenty of other information at https://pytorch.org/tutorials/

## Folder Structure

Below is an outline of the folde structure for the repo.

```
root
  -networks                   contains the training regimes for each network
      -__init__.py
      -mnist_cgan.py
      -mnist_classifier.py
      -mnist_gan.py
      -mnist_vae.py
  -torch_utils                folder of helper functions
      -__init__.py
      -dataset.py
      -torch_io.py
  -main.py                    the primary script which performs training
  -mnist                      example dataset
      -training
      -testing
```

## Running an example

To run an example

```
python3 main.py --network vae
```

# Tensorboard

To visualise the data we use the tensorboard_logger package, which can be found here https://github.com/TeamHG-Memex/tensorboard_logger
