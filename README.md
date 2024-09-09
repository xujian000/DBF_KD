# DBF_KD

## Usage

### Network Architecture

Our DBF_KD is implemented in `Net/DBF_KD.py`

### 🏊 Training

**1. Virtual Environment**

```
# create virtual environment
conda create -n cddfuse python=3.8.10
conda activate cddfuse
# select pytorch version yourself
# install cddfuse requirements
pip install -r requirements.txt
```

**2. Data Preparation**

Download the MSRS dataset from [this link](https://github.com/Linfeng-Tang/MSRS) and place it in the folder `'./MSRS_train/'`.

**3. Pre-Processing**

Run

```
python prepare_data.py
```

and the processed training dataset is in `'./data/MSRS_train_imgsize_128_stride_200.h5'`.

**4. Training**

Run

```
python train.py
```

and the trained model is available in `'./models/'`.

**5. Distillation**

Run

Copy the extra Medical Images for enhance the model, you can use any dataset you have
Put the MRI images in "./EnhanceDataset/vi" and the related CT/PET/SPECT in "./EnhanceDataset/ir"

and then run

```
python prepare_data_enhance.py
```

the processed extra dataset which used in distillation is in `'./data/Data4Enhance_imgsize_128_stride_200.h5'`.

```

and then run

python distill.py
```

and the trained model is available in `'./models/'`.

### 🏄 Testing

**1. Predistilled models**

Predistilled models are available in `'./models/DBF_KD.pth'`

**2. Test datasets**

The test datasets used in the paper have been stored in `'./test_img/MRI_CT'`, `'./test_img/MRI_PET'` and `'./test_img/MRI_SPECT'` for MIF.
