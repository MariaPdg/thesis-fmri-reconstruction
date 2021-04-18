# Image reconstruction from human brain activity 

## 3-stage training 

### Configs

* **Data config**

Specify path to save the results in ```data_config.py```:

```python
save_training_results = 'results/'
```

* **Training configs**

   - ```gan_config.py``` for vae/gan training (all stages, see below)
   - ```wae_config.py``` for wae/gan training (all stages, see below)
    

* Some important parameters:

  - Device for training:
    ```python
    device = 'cuda:4'  # cuda or cpu
    ```
  
  - If you want to continue training from the checkpoint specify the model name, e.g:
    ```python
    pretrained_gan = None  # e.g. 'gan_20210203-173210' 
    load_epoch = 395
    evaluate = False
    ```
    
  - If you want to evaluate only specify ```evaluate=True```
    
  - Image resolution and latent space dimension:
    
    ```python
    image_size = 64
    latent_dim = 128
    ```
    
  - Maen and std for image standardization:
    
    ```python
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    ```

### 1. Dual-VAE/GAN 

#### Stage I

1. Specify parameters in ```configs/gan_config.py```

2. Run training

    ```python
    python3 train/train_vgan_stage1.py -i [user path 1] -o [user path 2] -l [path to logs]
    ```
      Flags:
      * ``` -i [user path 1]``` user path where datasets are located 
      * ``` -o [user path 2]``` user path to save the results
      * ``` -l [path to logs]``` path to save logs


#### Stage II

1. Specify the model from stage I in ```configs/gan_config.py```.  
   
   Model name includes the name and epoch number, e.g.:
    ```python
    # Trained model for stage II (model from stage I)
    decoder_weights = ['gan_20210127-012348', 90]  # model, epoch
    ```

2. Run training
    ```python
    python3 train/train_vgan_stage2.py -i [user path 1] -o [user path 2] -l [path to logs]
    ```
   

#### Stage III

1. Specify the model from stage II in ```configs/gan_config.py```. 
   
    Model name includes the name and epoch number, e.g.:
    ```python
    # Trained model for stage III (model from stage II)
    cog_encoder_weights = ['gan_cog_2st_20210223-224153', 395]
    ```

2. Run training
    ```python
    python3 tn/train_vgan_stage3.py -i [user path 1] -o [user path 2] -l [path to logs]
    ```
   
### 2. WAE/GAN


#### Stage I

1. Specify parameters  in ```configs/wae_config.py```:
2. Run training: 

    ```python
    python3 train/train_wae_stage1.py -i [user path 1] -o [user path 2] -l [path to logs]
    ```
     Flags:
  * ``` -i [user path 1]``` user path where datasets are located 
  * ``` -o [user path 2]``` user path to save the results
  * ``` -l [path to logs]``` path to save logs



#### Stage II
    
1. Specify the model from Stage I in ```configs/wae_config.py```:

    ```python
    # [model, epoch]
    decoder_weights = ['wae_gan_20210222-164403', 398]   # latent dim = 128
    ```
2. Run training
    ```python
    python3 train/train_wae_stage2.py -i [user path 1] -o [user path 2] -l [path to logs]
    ```
   
#### Stage III
    
1. Specify the model from Stage I in ```configs/wae_config.py```:

    ```python
    # [model, epoch]
    cog_encoder_weights = ['waegan_cog_20210223-225621', 395]
    ```
2. Run training
    ```python
    python3 train/train_wae_stage3.py -i [user path 1] -o [user path 2] -l [path to logs]
    ```
   
### 3. WAE/Dual-GAN for Stage I

#### Stage I

1. Specify parameters in ```configs/gan_config.py```

2. Run training

    ```python
    python3 train/wae_vgan_stage1.py -i [user path 1] -o [user path 2] -l [path to logs]
    ```
      Flags:
      * ``` -i [user path 1]``` user path where datasets are located 
      * ``` -o [user path 2]``` user path to save the results
      * ``` -l [path to logs]``` path to save logs



