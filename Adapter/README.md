## SAM-Adapter code
We thank the paper SAM Fails to Segment Anything? – SAM-Adapter: Adapting SAM in Underperformed Scenes:Camouflage, Shadow, and More for providing code references.

## Installation
This code was implemented with Python 3.8 and PyTorch 1.13.0. You can install all the requirements via:
```bash
pip install -r requirements.txt
```

## Train
1. Download the dataset and put it in ./load.
2. Download the pre-trained [SAM(Segment Anything)](https://github.com/facebookresearch/segment-anything) and put it in ./pretrained.
3. Training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch train.py --nnodes 1 --nproc_per_node 4 --config [CONFIG_PATH]
```
!Please note that the SAM model consume much memory. We use A100 graphics card for training. If you encounter the memory issue, please try to use graphics cards with larger memory!


```bash
!torchrun train.py --config configs/demo.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 loadddptrain.py --config configs/demo.yaml
```

## Test
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH]
```

## Dataset


