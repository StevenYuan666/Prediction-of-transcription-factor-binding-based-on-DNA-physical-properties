# Prediction-of-transcription-factor-binding-based-on-DNA-physical-properties

## Introduction
Experiments such as Chip-Seq can identify a list of DNA regions bound by a given
transcription factor. Combined with a computational scan for the TF’s position-
weight matrix, this can be used to identify sites that are occupied by the TF in the
cell type and conditions where the experiments were made. We can also identify a
set of DNA sequences that, based on their sequence, look like they should be bound,
but that in reality are not. The goal of the project is to determine whether the sites
that are bound (positive examples) can be distinguished from those that are not
(negative example) based on the predicted structural properties of the sequence.
The project will involve (1) Identifying a set of bound and non-bound DNA
sequences for a given TF based on existing experimental data; (2) Calculating the
DNA physical properties of each sequence; (3) Training a machine learning classifier
to distinguish between bound and unbound sites.

## Install Dependencies
``` Shell
$ conda create --name CompBio python=3.8
$ conda activate CompBio
$ pip install -r requirements.txt
```


## Maintain Dependencies
All required packages should be listed clearly in the `requiremens.txt` documents.
To quickly update the newly installed packages, run the following command:
``` Shell
$ pip freeze > requirements.txt
```


## Continuous Integration
To keep the code clean and consistent, we use [isort](https://pycqa.github.io/isort/), [black](https://github.com/psf/black), and [flake8](https://flake8.pycqa.org/en/latest/) to check the code style.

Please run the following commands before you commit your code and push it to the remote repository:
```Shell
$ isort . --profile=black
$ black .
$ flake8 .
```
and make sure there is no error or warning.