import multiprocessing

#algo='clear'
eval_mode=False
file_path = r'C:\Users\phili\Downloads\10428077\LTR'
project_name = 'AspectJ'
num_actors=4

if eval_mode:
    episode=1000
else:
    train_data_path = r'C:\Users\phili\Downloads\10428077\bug_localization\RL_Model\AspectJ.csv'
    episode=3200
#env_name="CartPole-v0"
#task_name='single_CartPole-v0'

