"# LLTP" 
环境准备
```
conda create --name lltp python=3.8 -y
conda activate lltp
```
```
conda install pytorch torchvision -c pytorch

pip install -U openmim
mim install mmengine
mim install 'mmcv>=2.0.0rc4'
mim install 'mmdet>=3.0.0'
```

```
cd mmdetection3d
pip install -v -e .
```

运行模型
```python
python tools/trian.py myconfig.py
```
