import subprocess
import threading
import os 

def run(commands, output_file="output.txt"):
    def run_command(command, output_file):

        with open(output_file, "a") as f:
            subprocess.run(" ".join(command), shell=True, stdout=f, stderr=f)

    threads = []
    for i,cmd in enumerate(commands):
        thread = threading.Thread(target=run_command, args=(cmd, output_file))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


commands = []
lr = 1e-3
rank =  10
gpu = 0
alpha = 20 
root_path = './data/' 
epoch = 10


datasets = ['imagenet_r', 'imagenet_r', 'imagenet_r', 'cifar100', 'sdomainet']
paths = [f'{root_path}/imagenet-r', f'{root_path}/imagenet-r', f'{root_path}/imagenet-r',f'{root_path}/cifar100', f'{root_path}/domain']
tasks = [20, 5, 10, 10, 5]



for i in range(len(datasets)):
    commands = []
    dataset, path, task = datasets[i], paths[i], tasks[i]
    
    a = f"CUDA_VISIBLE_DEVICES={gpu} python train_splitlora.py -d {dataset} \
        -m vit_base_patch16_224.augreg2_in21k_ft_in1k --head_dim_type task_classes --logit_type head_out\
        -b 256 --temperature 30 \
        --data_root {path} --seed 0\
        --rank {rank} --lr {lr} --num_tasks {task} --alpha {alpha} --epoch {epoch}".split(' ')
    commands.append(a)
    print(a)
    run(commands, output_file=f"{dataset}_{task}.txt")
