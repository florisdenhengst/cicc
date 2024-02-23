# CICC 
Code for conformal intent classification and clarification

## Requirements
CICC uses python 3.

## Installation & Prerequisites
Run ``pip install -r requirements.txt``.

## Data set preparations
Most datasets in this repository are loaded using the 🤗 ecosystem.

However, the 'mtod' dataset from the 2019 paper by Schuster et al. entitled "Cross-lingual transfer learning for multilingual task oriented dialog" requires some manual step.

First download the official [data release](https://fb.me/multilingual_task_oriented_data).
into ``data/mtod/`` and unzip it there, then:

    ```bash
    cd data/mtod/ # if you are not in this dir already
    convert_mtod.py
    ```

## Usage
Run ``jupyter notebook`` in the ``cicc`` root directory and open a notebook to
an example of using CICC.

Notebooks starting with ``cicc_`` contain demontrations  and evaluation of the CICC framework. 

Notebooks starting with ``pretrain_`` contain code for fine-tuning pre-trained models for intent classification and are not specific to CICC. These have been included for reproducibility.