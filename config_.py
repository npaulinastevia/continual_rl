import multiprocessing

#algo='clear'
eval_mode=False
file_path = r'/scratch/f/foutsekh/nstevia/LTR'
project_name = 'AspectJ'
num_actors=4

if eval_mode:
    episode=1000
else:
    train_data_path = r'/scratch/f/foutsekh/nstevia/bug_localization/org.aspectj/AspectJ_train_test_before_fix_after_bug_report.csv'
    episode=3200
#env_name="CartPole-v0"
#task_name='single_CartPole-v0'

