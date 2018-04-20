# Generative Adversarial Networks (GANs)

### Technology Requirements:

1. Language: Python 3.6
2. Libraries: TensorFlow - GPU 1.4.0
3. GPU: nVidia GTX 1080
4. Python packages: pandas, numpy and datetime

### Instructions to run the code:

1. Git clone ``` https://github.ncsu.edu/mraveen/CSC522-GAN.git ```
2. ``` cd CSC522-GAN/gan_mnist/ ```
3. Install all the python packages and tensorflow with ``` pip install <package-name> ```
4. Run ``` python training.py ```
5. Now open a new terminal and run tensorboard to visualize the GAN performance ``` tensorboard --logdir=/home/<user>/CSC522-GAN/gan_mnist/tensorboard_mnist/ ```
6. Open the tensorboard dashboard at http://localhost:6006
7. You can find the scalars representations of discriminator (real and fake) ,and generator losses and the generated images as the program runs.

#### If using a GPU (Instructions based on ARC cluster provided by NCSU)

1. Login to the cluster provided.
2. Select a GPU by looking at the options ``` sinfo ```. Select a GPU greater than gtx680 as tensorflow needs that.
3. Example command to select the GPU ``` srun -n 16 -N 1 -p gtx1080 --pty /bin/bash ``` which selects the gtx1080 with 16 cores.
4. Perform 1 to 4 from previous instructions.
5. Now, extract the tensorboard event data every few minutes from the remote host to your local machine using scp or Winscp. A cronjob is advised.
6. Perform 5 to 7 from previous instructions.






