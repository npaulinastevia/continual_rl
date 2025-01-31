import logging
import tempfile
import types
from pathlib import Path

import gym
import numpy as np
import random
import torch
import os
import tempfile

from .Environment import LTREnvV2
from .cartpole_v0 import CartPoleEnv
from .nscartpole_v0 import NSCartPoleV0
from .nscartpole_v2 import NSCartPoleV2
from .nscartpole_v1 import NSCartPoleV1

class Utils(object):


    @classmethod
    def create_logger(cls, file_path):
        """
        The path must be unique to the logger you're creating, otherwise you're grabbing an existing logger.
        """
        logger = logging.getLogger(file_path)

        # Since getLogger will always retrieve the same logger, we need to make sure we don't add many duplicate handlers
        # Check if we've set this up before by seeing if handlers already exist
        if len(logger.handlers) == 0:
            formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.DEBUG)

        return logger

    @classmethod
    def make_env(cls, env_spec, create_seed=False, seed_to_set=None, max_tries=2,model_flags=None):
        """
        Seeding is done at the time of environment creation partially to make sure that every env gets its own seed.
        If you seed before forking processes, the processes will all be seeded the same way, which is generally
        undesirable.
        If create_seed is False and seed_to_set is None, no seed will be set at all.
        :param env_spec: The specification used to create an env. Can be a name (for OpenAI gym envs) or a lambda,
        which will get called each time creation is desired.
        :param create_seed: If True, a new seed will be created and set (for numpy, random, torch, and the env)
        :param seed_to_set: If not None, a seed will be set to the same locations as create_seed
        :return: (env, seed): The environment and the seed that was set (None if no seed was set)
        """
        seed = None
        make_env_tries = 0
        env = None
        import config_
        import time
        id_txt = str(int(time.time()))
        #reg_path='/scratch/nstevia/regression_model_develop/Aspectj/final_logistic_regression_model.pt'
        #test_AspectJ_train_before_fix_aft4er_bug_report.csv' AspectJ_test_for_mining_before.csv
        #path1='/scratch/nstevia/continual_rl/AspectJ_bug_loc_experiments_with_metric/train_AspectJ_train_before_fix_aft4er_bug_report.csv'#/scratch/nstevia/continual_rl/train_AspectJ_train_before_fix_aft4er_bug_report.csv'
        #path_test='/scratch/nstevia/continual_rl/AspectJ_bug_loc_experiments_with_metric/test_AspectJ_train_before_fix_aft4er_bug_report.csv'#'/scratch/nstevia/continual_rl/AspectJ_bug_loc_experiments_with_metric/AspectJ_test_for_mining_before.csv'#'/scratch/nstevia/continual_rl/AspectJ_test_withcommit.csv'#AspectJ_test_withcommit.csv'  test_AspectJ_train_before_fix_aft4er_bug_report.csv
        #path2='/scratch/nstevia/continual_rl/AspectJ_bug_loc_experiments_with_metric/AspectJ_train_for_mining_before.csv'
        #f = open('results_cart_trainCRL_' + env_spec + '.txt', 'a+')
        #f.write('Environments,Algorithms,x1,reward,steps,time,episodes,task_id,done' + '\n')
        #f.close()
        #f = open('results_cart_testCRL_' + env_spec + '.txt', 'a+')
        #f.write('Environments,Algorithms,x1,reward,steps,time,episodes,env_id,task_id,done' + '\n') train_with_metrics_SWT_dataset_baseline.csv
        #f.close()
        #train_data_path = config_.train_data_path
        #1 SWT, 2 JDT, 3 BIRT, 4 ECLISPE, 5 tomcat
        file_path = config_.file_path
        Path(file_path).mkdir(parents=True, exist_ok=True)
        print(env_spec,'envspec')

        if env_spec[0]=='0':
            non_original = False
        else:
            non_original = True
        if env_spec[1]=='1':
            project_name = 'SWT'
            reg_path='/scratch/nstevia/regression_model_develop/swt/final_logistic_regression_model.pt'
            #test_AspectJ_train_before_fix_aft4er_bug_report.csv' AspectJ_test_for_mining_before.csv
            path1='/scratch/nstevia/bug_rep_paulina/train_all_non_baseline_with_metrics_SWT_dataset_baseline.csv'#/scratch/nstevia/continual_rl/train_AspectJ_train_before_fix_aft4er_bug_report.csv'
            if env_spec[2]=='0':
                path_test='/scratch/nstevia/bug_rep_paulina/test_with_metrics_SWT_dataset_baseline.csv'#'/scratch/nstevia/continual_rl/AspectJ_bug_loc_experiments_with_metric/AspectJ_test_for_mining_before.csv'#'/scratch/nstevia/continual_rl/AspectJ_test_withcommit.csv'#AspectJ_test_withcommit.csv'  test_AspectJ_train_before_fix_aft4er_bug_report.csv
            else:
                path_test = '/scratch/nstevia/bug_rep_paulina/test_all_non_baseline_with_metrics_SWT_dataset_baseline.csv'
            path2='/scratch/nstevia/bug_rep_paulina/train_with_metrics_SWT_dataset_baseline.csv'#'/scratch/nstevia/continual_rl/AspectJ_train.csv'
            mpath ='/scratch/nstevia/bug_localization/micro_codebert'# '/scratch/f/foutsekh/nstevia/bug_localization/micro_codebert'#'/home/paulina/Downloads/micro_codebert'#'/scratch/f/foutsekh/nstevia/continual_rl/continual_rl/utils/micro_codebert'
        elif env_spec[1] == '2':
                project_name = 'JDT'
                reg_path = '/scratch/nstevia/regression_model_develop/jdt/final_logistic_regression_model.pt'
                # test_AspectJ_train_before_fix_aft4er_bug_report.csv' AspectJ_test_for_mining_before.csv
                path1 = '/scratch/nstevia/bug_rep_paulina/train_all_non_baseline_with_metrics_JDT_dataset_baseline.csv'  # /scratch/nstevia/continual_rl/train_AspectJ_train_before_fix_aft4er_bug_report.csv'
                if env_spec[2] == '0':
                    path_test = '/scratch/nstevia/bug_rep_paulina/test_with_metrics_JDT_dataset_baseline.csv'  # '/scratch/nstevia/continual_rl/AspectJ_bug_loc_experiments_with_metric/AspectJ_test_for_mining_before.csv'#'/scratch/nstevia/continual_rl/AspectJ_test_withcommit.csv'#AspectJ_test_withcommit.csv'  test_AspectJ_train_before_fix_aft4er_bug_report.csv
                else:
                    path_test = '/scratch/nstevia/bug_rep_paulina/test_all_non_baseline_with_metrics_JDT_dataset_baseline.csv'
                path2 = '/scratch/nstevia/bug_rep_paulina/train_with_metrics_JDT_dataset_baseline.csv'  # '/scratch/nstevia/continual_rl/AspectJ_train.csv'
                mpath = '/scratch/nstevia/bug_localization/micro_codebert'  #
        elif env_spec[1] == '3':
                project_name = 'Birt'
                reg_path = '/scratch/nstevia/regression_model_develop/birt/final_logistic_regression_model.pt'
                # test_AspectJ_train_before_fix_aft4er_bug_report.csv' AspectJ_test_for_mining_before.csv
                path1 = '/scratch/nstevia/bug_rep_paulina/train_all_non_baseline_with_metrics_Birt_dataset_baseline.csv'  # /scratch/nstevia/continual_rl/train_AspectJ_train_before_fix_aft4er_bug_report.csv'
                if env_spec[2] == '0':
                    path_test = '/scratch/nstevia/bug_rep_paulina/test_with_metrics_Birt_dataset_baseline.csv'  # '/scratch/nstevia/continual_rl/AspectJ_bug_loc_experiments_with_metric/AspectJ_test_for_mining_before.csv'#'/scratch/nstevia/continual_rl/AspectJ_test_withcommit.csv'#AspectJ_test_withcommit.csv'  test_AspectJ_train_before_fix_aft4er_bug_report.csv
                else:
                    path_test = '/scratch/nstevia/bug_rep_paulina/test_all_non_baseline_with_metrics_Birt_dataset_baseline.csv'
                path2 = '/scratch/nstevia/bug_rep_paulina/train_with_metrics_Birt_dataset_baseline.csv'  # '/scratch/nstevia/continual_rl/AspectJ_train.csv'
                mpath = '/scratch/nstevia/bug_localization/micro_codebert'  #
        elif env_spec[1] == '4':
                project_name = 'Eclipse_Platform_UI'
                reg_path = '/scratch/nstevia/regression_model_develop/eclipse/final_logistic_regression_model.pt'
                # test_AspectJ_train_before_fix_aft4er_bug_report.csv' AspectJ_test_for_mining_before.csv
                path1 = '/scratch/nstevia/bug_rep_paulina/train_all_non_baseline_with_metrics_Eclipse_Platform_UI_dataset_baseline.csv'  # /scratch/nstevia/continual_rl/train_AspectJ_train_before_fix_aft4er_bug_report.csv'
                if env_spec[2] == '0':
                    path_test = '/scratch/nstevia/bug_rep_paulina/test_with_metrics_Eclipse_Platform_UI_dataset_baseline.csv'  # '/scratch/nstevia/continual_rl/AspectJ_bug_loc_experiments_with_metric/AspectJ_test_for_mining_before.csv'#'/scratch/nstevia/continual_rl/AspectJ_test_withcommit.csv'#AspectJ_test_withcommit.csv'  test_AspectJ_train_before_fix_aft4er_bug_report.csv
                else:
                    path_test = '/scratch/nstevia/bug_rep_paulina/test_all_non_baseline_with_metrics_Eclipse_Platform_UI_dataset_baseline.csv'
                path2 = '/scratch/nstevia/bug_rep_paulina/train_with_metrics_Eclipse_Platform_UI_dataset_baseline.csv'  # '/scratch/nstevia/continual_rl/AspectJ_train.csv'
                mpath = '/scratch/nstevia/bug_localization/micro_codebert'  #
        elif env_spec[1] == '5':
                project_name = 'Tomcat'
                reg_path = '/scratch/nstevia/regression_model_develop/tomcat/final_logistic_regression_model.pt'
                # test_AspectJ_train_before_fix_aft4er_bug_report.csv' AspectJ_test_for_mining_before.csv
                path1 = '/scratch/nstevia/bug_rep_paulina/train_all_non_baseline_with_metrics_Tomcat_dataset_baseline.csv'  # /scratch/nstevia/continual_rl/train_AspectJ_train_before_fix_aft4er_bug_report.csv'
                if env_spec[2] == '0':
                    path_test = '/scratch/nstevia/bug_rep_paulina/test_with_metrics_Tomcat_dataset_baseline.csv'  # '/scratch/nstevia/continual_rl/AspectJ_bug_loc_experiments_with_metric/AspectJ_test_for_mining_before.csv'#'/scratch/nstevia/continual_rl/AspectJ_test_withcommit.csv'#AspectJ_test_withcommit.csv'  test_AspectJ_train_before_fix_aft4er_bug_report.csv
                else:
                    path_test = '/scratch/nstevia/bug_rep_paulina/test_all_non_baseline_with_metrics_Tomcat_dataset_baseline.csv'
                path2 = '/scratch/nstevia/bug_rep_paulina/train_with_metrics_Tomcat_dataset_baseline.csv'  # '/scratch/nstevia/continual_rl/AspectJ_train.csv'
                mpath = '/scratch/nstevia/bug_localization/micro_codebert'  #
        #env_spec=env_spec[4:]
        fi = open(env_spec+'.txt', 'a+')
        fi.write(path_test+','+project_name + '\n')
        fi.close()
        while env is None:
            try:
                if isinstance(env_spec, types.LambdaType):
                    env = env_spec()
                else:
                    if env_spec=='CartPole-v0':
                        env=CartPoleEnv()

                    elif env_spec=='NSCartPole-v0':
                        env=NSCartPoleV0()

                    elif env_spec=='NSCartPole-v1':
                        env=NSCartPoleV1()
                    elif env_spec=='NSCartPole-v2':
                        env=NSCartPoleV2()
                    elif "bug_log1" in env_spec:#env_spec=="bug_log1":

                        env=LTREnvV2(data_path=path1, model_path=mpath,
                            tokenizer_path=mpath, action_space_dim=31, report_count=100, max_len=512,
                        use_gpu=True, caching=True, file_path=file_path, project_list=[project_name],model_flags=model_flags,non_original=non_original,reg_path=reg_path)
                    elif "bug_log_test" in env_spec:

                        env=LTREnvV2(data_path=path_test, model_path=mpath,
                            tokenizer_path=mpath, action_space_dim=31, report_count=None, max_len=512,
                        use_gpu=True, caching=True, file_path=file_path, project_list=[project_name,env_spec],test_env=True,model_flags=model_flags,non_original=non_original,reg_path=reg_path)
                        #env=LTREnvV2(data_path=test_data_path, model_path=mpath,  # data_path=file_path + test_data_path
                        #tokenizer_path=mpath, action_space_dim=31, report_count=None, max_len=512,
                        #use_gpu=False, caching=True, file_path=file_path, project_list=[project_name], test_env=True, estimate=options.estimate)
                    elif "bug_log2" in env_spec:
                        env=LTREnvV2(data_path=path2, model_path=mpath,
                            tokenizer_path=mpath, action_space_dim=31, report_count=100, max_len=512,
                        use_gpu=True, caching=True, file_path=file_path, project_list=[project_name],model_flags=model_flags,non_original=non_original,reg_path=reg_path)
                    else:
                        env = gym.make(env_spec)
            except Exception as e:
                make_env_tries += 1
                if make_env_tries > max_tries:
                    raise e

        if create_seed or seed_to_set is not None:
            assert not (create_seed and seed_to_set is not None), \
                "If create_seed is True and a seed_to_set is specified, it is unclear which is desired."
            seed = cls.seed(env, seed=seed_to_set)
        print(env_spec)

        return env, seed

    @classmethod
    def seed(cls, env=None, seed=None):
        # Use the operating system to generate a seed for us, as it is not useful to seed a randomizer with itself
        # (Such a seed would be undesirably the same across forked processes.)
        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder="little")

        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        # In theory we should be able to call torch.seed, but that breaks on all but the most recent builds of pytorch.
        # https://github.com/pytorch/pytorch/issues/33546
        # So in the meantime do it manually. Tracked by issue 52
        # torch.seed()
        torch.manual_seed(seed)

        if env is not None:
            try:
                env.seed(seed)
            except:
                print("Environment does not support seeding")

        return seed

    @classmethod
    def get_max_discrete_action_space(self, action_spaces):
        max_action_space = None
        for action_space in action_spaces.values():
            if max_action_space is None or action_space.n > max_action_space.n:
                max_action_space = action_space
        return max_action_space

    @classmethod
    def create_file_backed_tensor(self, file_path, shape, dtype, shared=True, permanent_file_name=None):
        """
        If permanent_file_name is None, a temporary file will be created instead
        """
        # Enable both torch dtypes and numpy dtypes
        numpy_to_torch_dtype_dict = {
            np.bool: torch.bool,
            np.uint8: torch.uint8,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.complex64: torch.complex64,
            np.complex128: torch.complex128
        }

        # Convert to the torch dtype, if it's numpy
        if dtype in numpy_to_torch_dtype_dict:
            dtype = numpy_to_torch_dtype_dict[dtype]

        if permanent_file_name is None:
            file_handle = tempfile.NamedTemporaryFile(dir=file_path)
            file_name = file_handle.name
            print(f"Creating temporary file backed tensor: {file_name}")
        else:
            file_name = os.path.join(file_path, permanent_file_name)
            file_handle = None
            print(f"Creating or loading permanent file backed tensor: {file_name}")

        size = 1
        for dim in shape:
            size *= dim

        storage_type = None
        tensor_type = None
        if dtype == torch.uint8:
            storage_type = torch.ByteStorage
            tensor_type = torch.ByteTensor
        elif dtype == torch.int32:
            storage_type = torch.IntStorage
            tensor_type = torch.IntTensor
        elif dtype == torch.int64:
            storage_type = torch.LongStorage
            tensor_type = torch.LongTensor
        elif dtype == torch.bool:
            storage_type = torch.BoolStorage
            tensor_type = torch.BoolTensor
        elif dtype == torch.float32:
            storage_type = torch.FloatStorage
            tensor_type = torch.FloatTensor

        shared_file_storage = storage_type.from_file(file_name, shared=shared, size=size)
        new_tensor = tensor_type(shared_file_storage).view(shape)

        return new_tensor, file_name, file_handle

    @classmethod
    def count_trainable_parameters(cls, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
