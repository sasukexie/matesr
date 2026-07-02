# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29, 2022/7/13, 2022/7/18
# @Author : Zihan Lin, Yupeng Hou, Gaowei Zhang, Lei Wang
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn, zgw15630559577@163.com, zxcptss@gmail.com

import argparse
import os
from recbole.utils import get_local_time

from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function
import math

# 超参数自动调优：Hyperopt 单机调优，Ray Tune 分布式调优
def hyperopt_tune(args):
    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    # in other case, max_evals needs to be set manually
    config_file_list = (args.config_files.strip().split(" ") if args.config_files else None)
    hp = HyperTuning(
        args,                    # 命令行参数
        objective_function,      # 目标函数
        algo="exhaustive",       # 使用穷举搜索算法
        early_stop=10,           # 连续10次无改进则停止
        max_evals=200,           # 最多评估200组参数
        params_file=args.params_file,  # 从文件读取超参数空间
        fixed_config_file_list=config_file_list,  # 固定配置文件
        display_file=args.display_file,  # 可视化输出文件
    )
    hp.run()
    hp.export_result(output_file=args.output_file)
    print("best params: ", hp.best_params)
    print("best result: ")
    print(hp.params2result[hp.params2str(hp.best_params)])


def ray_tune(args):
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    config_file_list = (
        [os.path.join(os.getcwd(), file) for file in config_file_list]
        if args.config_files
        else None
    )
    params_file = (
        os.path.join(os.getcwd(), args.params_file) if args.params_file else None
    )
    ray.init()
    tune.register_trainable("train_func", objective_function)
    config = {}
    with open(params_file, "r") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            para_list = line.split(" ")
            if len(para_list) < 3:
                continue
            para_name, para_type, para_value = (
                para_list[0],
                para_list[1],
                "".join(para_list[2:]),
            )
            if para_type == "choice":
                para_value = eval(para_value)
                config[para_name] = tune.choice(para_value)
            elif para_type == "uniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.uniform(float(low), float(high))
            elif para_type == "quniform":
                low, high, q = para_value.strip().split(",")
                config[para_name] = tune.quniform(float(low), float(high), float(q))
            elif para_type == "loguniform":
                low, high = para_value.strip().split(",")
                config[para_name] = tune.loguniform(
                    math.exp(float(low)), math.exp(float(high))
                )
            else:
                raise ValueError("Illegal param type [{}]".format(para_type))
    # choose different schedulers to use different tuning optimization algorithms
    # For details, please refer to Ray's official website https://docs.ray.io
    scheduler = ASHAScheduler(
        metric="hit@10", mode="max", max_t=10, grace_period=1, reduction_factor=2
    )

    local_dir = "./ray_log"
    result = tune.run(
        tune.with_parameters(objective_function, config_file_list=config_file_list),
        config=config,
        num_samples=5,
        log_to_file=args.output_file,
        scheduler=scheduler,
        local_dir=local_dir,
        resources_per_trial={"gpu": 1},
    )

    best_trial = result.get_best_trial("hit@10", "max", "last")
    print("best params: ", best_trial.config)
    print("best result: ", best_trial.last_result)


def process(model_name, dataset_name, time_str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=model_name, help="name of models")
    parser.add_argument("--dataset", type=str, default=dataset_name, help="name of datasets")
    parser.add_argument("--config_files", type=str, default='zone/common.yaml zone/hyper.yaml', help="fixed config files")
    parser.add_argument("--params_file", type=str, default=None, help="parameters file (auto-detect per dataset)")
    parser.add_argument("--output_file", type=str, default=f"log/{model_name}/R_{dataset_name}_{time_str}.result",
                        help="output file")
    parser.add_argument("--display_file", type=str, default=f"log/{model_name}/V_{dataset_name}_{time_str}.result",
                        help="visualization file")
    parser.add_argument("--tool", type=str, default="Hyperopt", help="tuning tool")
    args, _ = parser.parse_known_args()

    # 按数据集选择专属参数文件，不存在则回退到通用文件
    if args.params_file is None:
        dataset_params = f"zone/hyper.{dataset_name}.params"
        fallback_params = "zone/hyper.params"
        if os.path.exists(dataset_params):
            args.params_file = dataset_params
        else:
            args.params_file = fallback_params
            print(f"[WARN] {dataset_params} not found, using {fallback_params}")

    if args.tool == "Hyperopt":
        hyperopt_tune(args)
    elif args.tool == "Ray":
        ray_tune(args)
    else:
        raise ValueError(f"The tool [{args.tool}] should in ['Hyperopt', 'Ray']")

if __name__ == "__main__":
    model_name = 'MATESR'
    dataset_name_arr = ['ml-100k','Amazon_Books','movielens','RentTheRunway','netflix','ml-1m'] # ['steam','netflix','RentTheRunway','movielens','lfm1b-artists','ml-10m','mind','lfm1b-tracks']
    for dataset_name in dataset_name_arr:
        time_str = get_local_time()
        process(model_name, dataset_name, time_str)
