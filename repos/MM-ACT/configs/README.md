# Configs Description

## LIBERO
Our vanilla model is trained exclusively using the [mmact_libero_action.yaml](mmact_libero_action.yaml) script. 
Please complete the model and dataset paths within the script and adjust the relevant parameters as needed. 
Note that ```batch_size``` refers to the batch size per device. 
You should adjust the effective global batch size by 
coordinating ```batch_size``` with ```gradient_accumulation_steps``` in your accelerate_configs.

For multimodal training pipeline used in LIBERO-Long, 
we first perform MMU training with the re-mask decoding strategy using [mmact_libero_mmu.yaml](mmact_libero_mmu.yaml). 
We then take the checkpoint trained for 1 epoch and continue training using [mmact_libero_mix.yaml](mmact_libero_mix.yaml). 
Note that in [mmact_libero_mix.yaml](mmact_libero_mix.yaml), ```batch_size``` defines the number of "context" per device. 
The total data count per device is calculated as: ```batch_size``` × number of training modalities. 
Please refer to our paper for the specific number of training steps to reproduce our training processes.

## RoboTwin
For RoboTwin, the script [mmact_robotwin_mix.yaml](mmact_robotwin_mix.yaml) allows for flexible training with any combination of the three modalities and two decoding strategies.
To customize the modality mix, set ```co_action```, ```co_mmu```, or ```co_t2i``` to True to include them in the training. 

To adjust the decoding strategies, 
modify ```predict_all_tokens``` (for action), ```mmu_eps``` (for text) and ```predict_all_image_tokens``` (for image)
following code annotations. The total batch size is alse calculated as: 
```batch_size``` × number of training modalities × number of devices.
