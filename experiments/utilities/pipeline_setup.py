
import os
import argparse
import bz2
import dill as pickle


def get_task_arr(out_dir):
    tasks_file = open(os.path.join(out_dir, 'setup', "tasks.txt"), 'r')
    tasks_list = tasks_file.readlines()
    tasks_file.close()

    return [tasks.strip().split(' ') for tasks in tasks_list]


def tasks_files(wildcards):
    return [os.path.join(wildcards.TMPDIR, 'output',
                         "out__cv-{}_task-{}.p".format(cv_id, task_id))
            for cv_id in range(40) for task_id in wildcards.tasks.split('-')]


def get_task_count(out_dir):
    task_count = 1
 
    with open(os.path.join(out_dir, 'setup', "tasks.txt"), 'r') as f:
        task_list = f.readline().strip()
 
        while task_list:
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
    parser.add_argument('time_max', type=int)
    parser.add_argument('--task_size', type=float, default=1)
    args = parser.parse_args()

    with open(os.path.join(args.out_dir, 'setup',
                           "muts-count.txt"), 'r') as f:
        muts_count = int(f.readline())

    with bz2.BZ2File(os.path.join(args.out_dir, 'setup',
                                  "cohort-data.p.gz"), 'r') as f:
        samp_count = len(pickle.load(f).get_samples())

    task_load = int((args.time_max * 4513)
                    / (args.task_size * samp_count ** 1.31))
    task_count = (muts_count - 1) // task_load + 1
    merge_count = task_count // 2 + 1

    task_list = tuple(range(task_count))
    task_arr = []

    for i in range(merge_count):
        task_arr += [
            "{}\n".format(
                ' '.join([str(tsk) for tsk in task_list[i::merge_count]]))
            ]

    task_file = open(os.path.join(args.out_dir, 'setup', "tasks.txt"), "w")
    task_file.writelines(task_arr)
    task_file.close()


if __name__ == '__main__':
    main()

