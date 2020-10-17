![to see the whole pdf report, please click here](https://github.com/yanghaoxiang7/Yang-Haoxiang-s-First-Research-Turn/blob/master/report1.pdf)

# Yang Haoxiang's First Research Turn
Codes and Experiment results

The GAN Face Recognition model is borrowed from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html/.

The zi2zi model is borrowed from https://github.com/kaonashi-tyc/zi2zi.

The code has beed testified under the following environment:

Python >= 3.6

CUDA >= 10.0

GPU memory >= 2.0GB

<1> mnist_classification.py

<2> mnist_generation.py

<3> celebA_generation.py

<4> zi2zi_master

In this section please follow these step to run the code:

(1) check your Python and CUDA version

(2) type in: （you should first creat a directory ./sample_dir）

python font2img.py --src_font=simfang.ttf --dst_font=simkai.ttf --charset=CN --sample_count=1000 --sample_dir=./sample_dir --label=0 --filter=1 --shuffle=1

to transfer characters to images (and prepare to transfer simfang.ttf to simkai.ttf)

(3) copy photos in ./sample_dir to ./image_directories and type in: （you should first creat a directory ./image_directories and ./binary_save_directory）

python package.py --dir=./image_directories --save_dir=./binary_save_directory --split_ratio=0.5

(4) copy the .obj files in ./binary_save_directory to ./experiment/data, and run

python train.py --experiment_dir=./experiment

and you will see the results in ./exeperiment/sample
