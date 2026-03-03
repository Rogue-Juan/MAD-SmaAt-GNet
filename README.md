# MAD-SmaAt-GNet
Code for the thesis "MAD-SmaAt-GNet: A multimodal advection-directed network for precipitation nowcasting".
<img width="2344" height="2341" alt="mad-smaat-gnet" src="https://github.com/user-attachments/assets/ff8122ec-7f00-466c-bd91-1db2e1eb105b" />
The MAD-SmaAt-GNet model can be found in mad_smaat_gnet/models/[MAD_SmaAt_GNet.py](https://github.com/Rogue-Juan/MAD-SmaAt-GNet/blob/main/mad_smaat_gnet/models/MAD_SmaAt_GNet.py).

## Installing dependencies
A snapshot of the used packages with which the models were trained is given in [requirements.txt](https://github.com/Rogue-Juan/MAD-SmaAt-GNet/blob/main/requirements.txt). To reproduce the results, one can install the dependencies as follows:
```
pip install -r requirements.txt
```
Note that the models were trained with a GPU that was CUDA-enabled; as such, PyTorch was installed with CUDA enabled. If you do not want this, replace ``torch==2.8.0+cuda126`` with ``torch==2.8.0`` in your requirements.txt. The training and testing scripts check whether CUDA is available and do not need to be changed. However, the data loader has ``pin_memory=True`` and ``num_workers=1`` which are optimal for CUDA. Change these in the code if necessary.

## Training
The training script for the models is given in [madsmaat_train_model.py](https://github.com/Rogue-Juan/MAD-SmaAt-GNet/blob/main/madsmaat_train_model.py).
NOTE THAT the path to the dataset and the ``default_save_path`` needs to be set by the user manually as well as the path to a pre-trained evo-net which is used in the models MAD-SmaAt-GNet and SmaAt-UNet with Evo-Net. The model to be trained can be set by changing ``args.model_name`` to the name of that model. The hyperparameter values used for the training of the models are the values listed in the script. Note that the data are normalised and denormalised in the training script.

## Testing
The testing script for the models is given in [madsmaat_test_models.py](https://github.com/Rogue-Juan/MAD-SmaAt-GNet/blob/main/madsmaat_test_model.py).
NOTE THAT the path to the dataset, the path to the checkpoint file of the model, and the path for the JSON file of the results need to set manually by the user. Similar to the training script, the user can specify which model to test by changing ``args.model_name`` to the correct model. The hyperparameter values used for the pre-trained models are the values listed in the script. Note that the data are normalised and denormalised in the training script.

Additionally, the results can be obtained per time step, i.e. per image, with the [test_model_per_img.py](https://github.com/Rogue-Juan/MAD-SmaAt-GNet/blob/main/test_model_per_img.py). This script operates the same way as madsmaat_test_models.py but calculates the results per image.

## Plots
The plots of the paper were generated with the testing script for the model predictions and ground truth, and [scatterplots.py](https://github.com/Rogue-Juan/MAD-SmaAt-GNet/blob/main/scatterplots.py) for the plot of the MSE over time. The code for the plots of the cropping of an example rain image and the plots of example images of other weather variables are not on this repository and require the raw data from the [Royal Dutch Metereological Institute (KNMI)](https://www.knmi.nl/over-het-knmi/about). The scatterplots.py requires the JSON files of the results per image of the models.

## Data
The data used for training the models came from simulation data by the [HARMONIE model](https://english.knmidata.nl/open-data/harmonie) of the KNMI and the data were provided by the [KNMI](https://dataplatform.knmi.nl/). From these data, accumulated rain, temperature at 300 metres, air pressure, relative humidity, and U- and V-wind at 300 metres were selected and cropped to the geographical region of the Netherlands (and parts of the North Sea), bounded by the coordinates $`[50.84, 53.462]^{\circ}`$ N latitude, $`[3.182, 7.4]^{\circ}`$ E longitude. The accumulated rain over time was converted to mm/h by subtracting the previous rain image from the current rain image, for all rain images. Additionally, the data were filtered on the rain images such that only samples were added to the dataset (with corresponding images of other weather variables) if the first input rain images had at least 20% of its pixels with rain intensities greater than 0.1 mm/h.

The plot of the cropping of an example rain image is given below:
<img width="1100" height="443" alt="rain_cropping_figure" src="https://github.com/user-attachments/assets/578266fb-1e40-4a77-ad8c-9f817267bfca" />

The plot of the corresponding images of other weather variables is given here below:
<img width="1920" height="975" alt="other_weather_vars" src="https://github.com/user-attachments/assets/9520b2e8-e93f-44c6-bf36-36f77e837ff0" />

If you are interested in any of the data that was used in the paper, please write an email to [s.mehrkanoon@uu.nl](mailto:s.mehrkanoon@uu.nl).

The dataset consisted of 5,925 training samples and 1,883 test samples in the HDF5 format with file size 10.3 GB.

## Citation
```
@report{VANWONDEREN2025,
title={MAD-SmaAt-GNet: A multimodal advection-directed network for precipitation nowcasting},
year={2025},
author={Samuel van Wonderen and Siamak Mehrkanoon},
abstract={Present-day precipitation forecasting is often still done with numerical solvers for physical equations that require much computational time and do not use the bulks of available weather data. Deep-learning models have shown much potential in precipitation forecasting, especially for short time horizons (known as "nowcasting"), due to their computational efficiency. Among these models, convolutional neural networks (CNNs) excel in image-to-image tasks, even for image sequences. The SmaAt-UNet is a light-weight CNN model that performed well for precipitation nowcasting. In our paper, SmaAt-UNet was extended with an extra encoder for other weather variables and with a physics-based component, and the Multimodal Advection-Directed Small Attention G-Net model (MAD-SmaAt-GNet) was developed. It was shown that each extension separately improves the rain predictions and that these extensions jointly improve the predictions even more, with a reduction of 9.8\% in MSE compared to SmaAt-UNet for predicted sequences of 4 consecutive rain images up to 4 hours ahead. Furthermore, the experiments showed that the input of additional weather variables is most beneficial for short time horizons whereas the advection-based component improved predictions for both the short and long term.},
}
```
