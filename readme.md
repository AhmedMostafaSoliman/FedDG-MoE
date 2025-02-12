# FedDG-MoE: Federated Domain Generalization with Dynamic Mixture-of-Expert Adaptation

## Requirements

- Python 3.9.7
- numpy 1.20.3
- torch 1.11.0
- torchvision 0.12.0

## Dataset

Firstly create directory for log files and change the dataset path (`pacs_path`, `officehome_path` and `terrainc_path`) and log path (`log_count_path`) in configs/default.py.
Please download the datasets from the official links:

- [PACS](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)
- [OfficeHome](https://hemanthdv.github.io/officehome-dataset)
- [TerraInc](https://beerys.github.io/CaltechCameraTraps)
- [DomainNet](http://ai.bu.edu/M3SDA/)

## Training from scratch

Run the code
`
python algorithms/feddg_moe/train_officehome.py --test_domain p --lr 0.001 --batch_size 64 --comm 40 --model clip_moe
`

## Acknowledgement

Part of our code is borrowed from the following repositories.
- FedDG-GA [https://github.com/MediaBrain-SJTU/FedDG-GA]
- FACT [https://github.com/MediaBrain-SJTU/FACT]
- DomainBed [https://github.com/facebookresearch/DomainBed]
- FedNova [https://github.com/JYWa/FedNova]
- SCAFFOLD-PyTorch [https://github.com/KarhouTam/SCAFFOLD-PyTorch]
We thank to the authors for releasing their codes. Please also consider citing their works.
