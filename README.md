## RGBD Saliency Net

![Architecture](./figures/architecture.png)

This is the source code of our paper "Learning RGB-D Salient Object Detection using background enclosure, depth contrast, and top-down features".

Our code is implemented based on [ELDNet](https://github.com/gylee1103/ELDNet) which is RGB saliency detection system. We also make use of [gSLICr](https://github.com/carlren/gSLICr) in our system.

## Usage
- **Supported OS**: We tested our code on Ubuntu 14.04.

- **Dependencies**: Basically see [Caffe installation](http://caffe.berkeleyvision.org/install_apt.html). We tested our code on CUDA 8.0, OpenCV 3.0.0.

- **Installation**

  1. We added scripts to original caffe. Please build our version caffe using CMake:

    ```shell
    # execute these command at the root of this directory
    cd caffe && mkdir build && cd build
    cmake ..
    make -j8
    ```

  1. Adjust library paths in CMakeList.txt and build code for test.

    ```shell
    # execute these command at the root of this directory
    edit CMakeList.txt
    mkdir build && cd build
    cmake ..
    make
    ```

- **Run demo program**

  ```shell
  sh demo.sh
  ```
If you want to test NJUDS2000 dataset images, please use NJUDS2000.caffemodel.

## How to create fill and gap maps
  ```shell
    # execute these command at the root of this directory
    cd create_fill_gap && mkdir build && cd build
    cmake ..
    make
    # PLEASE EDIT create_fill_gap.sh TO FIT YOUR ENVIRONMENT
    sh create_fill_gap.sh
  ```

## Results in our paper

### All saliency map outputs are contained in a paper_results.zip file.

![results](./figures/results.png)

## Citing our work
Please kindly cite our work if it helps your research:

  ```shell
  @InProceedings{Shigematsu_2017_ICCV,
  author = {Shigematsu, Riku and Feng, David and You, Shaodi and Barnes, Nick},
  title = {Learning RGB-D Salient Object Detection Using Background Enclosure, Depth Contrast, and Top-Down Features},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV) Workshops},
  month = {Oct},
  year = {2017}
  }
  ```
