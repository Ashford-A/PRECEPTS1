
import os
import argparse
import bz2
import dill as pickle
import pipes
from math import ceil

DIV_LINE = "=====\n"


def get_task_arr(out_dir):
    tasks_file = open(os.path.join(out_dir, 'setup', "tasks.txt"), 'r')
    tasks_list = tasks_file.readlines()
    tasks_file.close()
    tsk_stop = tasks_list.index(DIV_LINE)

    return [tasks.strip().split(' ') for tasks in tasks_list[:tsk_stop]]


def tasks_files(wildcards):
    return [os.path.join(wildcards.TMPDIR, 'output',
                         "out__cv-{}_task-{}.p".format(cv_id, task_id))
            for cv_id in range(40) for task_id in wildcards.tasks.split('-')]


def get_task_count(out_dir):
    task_count = 1
 
    with open(os.path.join(out_dir, 'setup', "tasks.txt"), 'r') as f:
        task_list = f.readline().strip()
 
        while task_list != DIV_LINE.strip():
            task_count = max(task_count,
                             *[int(tsk) + 1 for tsk in task_list.split(' ')])
            task_list = f.readline().strip()

    return task_count


def main():
    parser = argparse.ArgumentParser(
        'setup_tasks',
        description="Figures out how to parallelize classification tasks."
        )

    parser.add_argument('out_dir', type=str)
    parser.add_argument('run_max', type=int)
    parser.add_argument('--merge_max', type=int)
    parser.add_argument('--task_size', type=float, default=1)
    args = parser.parse_args()

    with open(os.path.join(args.out_dir, 'setup',
                           "muts-count.txt"), 'r') as f:
        muts_count = int(f.readline())

    with bz2.BZ2File(os.path.join(args.out_dir, 'setup',
                                  "cohort-data.p.gz"), 'r') as f:
        samp_count = len(pickle.load(f).get_samples())

    task_load = args.run_max * 6137
    task_load //= 301 + args.task_size * samp_count ** 1.37
    task_count = int(((muts_count - 1) // task_load) + 1)
    task_size = muts_count // task_count

    if args.merge_max is None:
        merge_count = 1

    else:
        merge_load = (args.merge_max * 30701) // (53 + samp_count ** 1.47)
        merge_count = int(merge_load // task_size + 1)

    task_list = list(range(task_count))
    task_arr = [[] for _ in range(ceil(task_count / merge_count))]
    i = 0

    while task_list:
        tsk_indx = (len(task_list) - 1) // (i % len(task_list) + 1)
        task_arr[i] += [str(task_list.pop(tsk_indx))]
        i = (i + 1) % len(task_arr)

    merge_size = 1
    for i in range(len(task_arr)):
        task_arr[i] = "{}\n".format(' '.join(sorted(task_arr[i])))
        merge_size = max(merge_size, len(task_arr[i]))

    task_arr = sorted(task_arr) + [DIV_LINE]
    run_time = 1.07 * task_size * args.run_max / task_load
    task_arr += ["run_time={}\n".format(int(run_time) + 1)]
    merge_time = 1.03 * args.merge_max * task_size * merge_count / merge_load
    task_arr += ["merge_time={}\n".format(int(merge_time) + 1)]

    task_file = open(os.path.join(args.out_dir, 'setup', "tasks.txt"), "w")
    task_file.writelines(task_arr)
    task_file.close()


if __name__ == '__main__':
    main()

