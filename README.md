# Advantage Return computation in Off-Policy Policy Gradient

PyTorch implementation of ATD3 
TD3 dupicated from the orignal repository from the authors of [paper](https://arxiv.org/abs/1802.09477).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 0.4](https://github.com/pytorch/pytorch) and Python 2.7. 

### Usage
```
python main.py --env HalfCheetah-v1 --use_adv True
```

Hyper-parameters can be modified with different arguments to main.py. We include an implementation of DDPG (DDPG.py) for easy comparison of hyper-parameters with TD3, this is not the implementation of "Our DDPG" as used in the paper (see OurDDPG.py). 

