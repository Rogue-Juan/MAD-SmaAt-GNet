# MAD-SmaAt-GNet
Code for the thesis "MAD-SmaAt-GNet: A multimodal advection-directed network for precipitation nowcasting".
<img width="2354" height="2351" alt="mad-smaat-gnet" src="https://github.com/user-attachments/assets/247de4d5-8b5f-44fd-988e-66509c40a095" />
The MAD-SmaAt-GNet model can be found in mad_smaat_gnet/models/[MAD_SmaAt_GNet.py](https://github.com/Rogue-Juan/MAD-SmaAt-GNet/blob/main/mad_smaat_gnet/models/MAD_SmaAt_GNet.py).

## Installing dependencies
A snapshot of the used packages with which the models were trained is given in [requirements.txt](https://github.com/Rogue-Juan/MAD-SmaAt-GNet/blob/main/requirements.txt). To reproduce the results, one can install the dependencies as follows:
```pip install -r requirements.txt```
Note that the models were trained with a GPU that was CUDA-enabled; as such, PyTorch was installed with CUDA enabled. If you do not want this, replace ``torch==2.8.0+cuda126`` with ``torch==2.8.0`` in your requirements.txt. The training and testing scripts check whether CUDA is available and do not need to be changed. However, the data loader has ``pin_memory=True`` and ``num_workers=1`` which are optimal for CUDA. Change these in the code if necessary.

## Training
The training script for the models is given in [madsmaat_train_model.py](https://github.com/Rogue-Juan/MAD-SmaAt-GNet/blob/main/madsmaat_train_model.py).
NOTE THAT the path to the dataset and the ``default_save_path`` needs to be set by the user manually.
