# This file is a generic hyperparameter tuner, based on Approximately Exact Line Search https://arxiv.org/abs/2011.04721
# Assume we have a function called fun with an input parameter called hyper, and a given starting value of hyper.
import subprocess

# Wrap the function such that fun(hyper) returns the loss we want to minimize, and silences any output produced by the wrapped function
# The launch command is PYTHONPATH='.' python plenoxels/runners/run_pretrained_dict.py --config plenoxels/configs/reuse_dict.yaml
hyper_name = 'optim.lr'
hyper_guess = 2e-3
second_G = 'False'
expname = f'tuning_bignetsmallFfullreso128sample6x'
def fun(hyper):
    # process = subprocess.run(['python', 'plenoxels/runners/run_pretrained_dict.py', '--config', 'plenoxels/configs/reuse_dict.yaml', '--config-updates', 'expname', 'tuning', 'optim.num_epochs', '1', hyper_name, str(hyper)],
    #                         stdout=subprocess.PIPE,
    #                         universal_newlines=True)
    process = subprocess.run(['python', 'plenoxels/runners/run_single_scene.py', '--config', 'plenoxels/configs/sara_singlescene.yaml', '--config-updates', 'expname', expname, 'optim.num_epochs', '1', 'model.second_G', second_G, hyper_name, str(hyper)],
                            stdout=subprocess.PIPE,
                            universal_newlines=True)
    output = process.stdout
    lines = output.split('\n')
    best_psnr = 0
    for line in lines:
        if not ('Test PSNR' in line):
            continue
        print(line)
        psnr = float(line.split('Test PSNR=')[-1])
        if psnr > best_psnr:
            best_psnr = psnr
    return -best_psnr


# Based on https://github.com/modestyachts/AELS/blob/main/utils.py
def tune_approx_exact(fun, hyper_guess, beta=2/(1+5**0.5)):
    # First, evaluate the function with hyperparameter = 0 to get a baseline
    # print(f'evaluating with hyperparameter 1e-10 as a baseline')
    # curval = fun(1e-10)
    # print(f'with hyperparameter 1e-10, loss is {curval}')
    curval = -8
    print(f'using psnr=8 as a baseline of no learning')
    t = hyper_guess
    t_old = 1e-10
    f_old = curval
    print(f'evaluating with hyperparameter {t}')
    f_new = fun(t)
    print(f'with hyperparameter {t}, loss is {f_new}')
    f_first = f_new
    param = beta
    if f_new <= f_old:
        param = 1.0 / beta
    num_iters = 0
    count = 0
    while True:
        num_iters = num_iters + 1
        t_old = t
        t = t * param
        f_old = f_new
        print(f'evaluating with hyperparameter {t}')
        f_new = fun(t)
        print(f'with hyperparameter {t}, loss is {f_new}')
        if f_new > curval and param < 1: # Special case for nonconvex function to ensure function decrease
            continue
        if f_new == f_old: # Numerical precision can be a problem in flat places, so try increasing hyperparameter
            count = count + 1
        if count > 5: # But have limited patience for asymptotes/infima
            break
        if f_new > f_old or (f_new == f_old and param < 1):
            break
    # Handle special case where the function value decreased at t but increased at t/beta
    if count > 20 or (num_iters == 1 and param > 1):
        t_old, t = t, t_old
        param = beta
        f_old, f_new = f_new, f_old
        if count > 5: # Tried increasing the step size, but function was flat
            t = hyper_guess
            f_new = f_first
        count = 0
        while True:
            t_old = t
            t = t * param
            f_old = f_new
            print(f'evaluating with hyperparameter {t}')
            f_new = fun(t)
            print(f'with hyperparameter {t}, loss is {f_new}')
            if f_new == f_old:
                count = count + 1
            if count > 5: # Don't backtrack forever if the function is flat
                break
            if f_new > f_old: 
                break
    print(f'best loss was {f_old}, with parameter {t_old}')
    return t_old, f_old # The penultimate value is the best value seen


best_hyper, best_loss = tune_approx_exact(fun, hyper_guess)
