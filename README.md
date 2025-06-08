# Analyzing and evaluating different GCN-variants on the movie recommendation task
## 1. LightGCN
### Enviroment Requirement

`pip install -r requirements.txt`

### Training

`cd lightgcn/code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="ml-1m" --topks="[20]" --recdim=64`

## 2. SGL
## 3. UltraGCN
### Convert data format
`cd ultragcn/data/Movielens1M_m1 && python convert_data.py`
### Training
`cd ultragcn && python main.py --config_file ./config/ultragcn_movielens1m_m1.ini`
