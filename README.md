# MSc-Project
This repository will contain the codebase used to develop various machine/deep learning models to remove electrode motion noise from ECG signals.

Electrode Motion is notoriously difficult to remove from ECG using standard frequency domain Signal Procesing techniques, thus, Artificial Intelligence may provide a more efficient method for eliminating this type of noise.

General Overview of work

Dataset creation
A dataset of real ECG signals will be composed. These will then be overlayed with synthetic Electrode Motion noise of various SNR's, shapes etc. This will provide the basis of a training dataset.

Data Augmentation Techniques will be investigated using GAN's or CNN networks to synthesize further data, however the complexity of this is unknown.

Model development

Initially, simple Machine Learning algorithms will be trained and performance shown. It is not expected these will perform well given the complexity of the data and lack of gold standard.

Deep Learning will be employed as an alternative, performance should increase. Optimization will be performed to try and create a robust and efficent algorithm that can be readily deployed on cloud and edge devices.

Model validations
TBD
