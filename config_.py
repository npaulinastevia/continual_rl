import multiprocessing

#algo='clear'
eval_mode=False
file_path = r'./LTR'
project_name = 'AspectJ'
if eval_mode:
    episode=1000
else:
    train_data_path = r'./AspectJ_train.csv'
    episode=3200
#env_name="CartPole-v0"
#task_name='single_CartPole-v0'

