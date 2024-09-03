FC-MSAN:Sleep stage classification method based on sleep functional connectivity graph and multi-scale attention mechanism



These are source code and experimental setup for the ISRUC-S3 dataset.

## Datasets

We evaluate our model on the ISRUC-Sleep-S3 and ISRUC-Sleep-S1 dataset.

- The **ISRUC-Sleep-S3** dataset is available [here](https://sleeptight.isr.uc.pt/), and we provide the pipeline to run MKSTGCN on it.
- The **MASS-SS3** dataset is an open-access and collaborative database of laboratory-based polysomnography (PSG) recordings. Information on how to obtain it can be found [here](http://massdb.herokuapp.com/en/).

## How to run

- **1. Get Dataset:**
  
  You can download ISRUC-Sleep-S3 dataset by the following command, which will automatically download the raw data and extracted data to `./data/ISRUC_S3/`:

- **2. Data preparation:**

  To facilitate reading, we preprocess the dataset into a single .npz file:

  ```shell
  python preprocess.py
  ```
  
  In addition, distance based adjacency matrix is provided at `./data/ISRUC_S3/DistanceMatrix.npy`.
  
- **3. Configuration:**

  Write the config file in the format of the example.

  We provide a config file at `/config/ISRUC.config`

- **4. Feature extraction:**

  Run `python train_FC_MSAN.py` with -c and -g parameters. After this step, the features learned by a feature net will be stored.

  + -c: The configuration file.
  + -g: The number of the GPU to use. E.g.,`0`. Set this to`-1` if only CPU is used.

  ```shell
  python train_FC_MSAN.py -c ./config/ISRUC.config -g 0
  ```

- **5. Train MKSTGCN:**

  Run `python train_MKSTGCN.py` with -c and -g parameters. This step uses the extracted features directly. 

    ```shell
  python train_MKSTGCN.py -c ./config/ISRUC.config -g 0
    ```

- **6. Evaluate MKSTGCN:**

  Run `python evaluate_MKSTGCN.py` with -c and -g parameters.

    ```shell
  python evaluate_MKSTGCN.py -c ./config/ISRUC.config -g 0
    ```


> **Summary of commands to run:**
>
> ```shell
> python preprocess.py
> python train_FC_MSAN.py -c ./config/ISRUC.config -g 0
> python train_MKSTGCN.py -c ./config/ISRUC.config -g 0
> python evaluate_MKSTGCN.py -c ./config/ISRUC.config -g 0
> ```
>


- **Perference**
Multi-View Spatial-Temporal Graph Convolutional Networks With Domain Generalization for Sleep Stage Classification
