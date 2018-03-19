#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/chrono.hpp>
#include "caffe/layers/lowlevel_depth_layer.hpp"

using std::cout;
using std::endl;
using cv::Mat;

namespace caffe {
template <typename Dtype>
__global__ void CalculateGridFeatures(const int nthreads, const int grid_size,
    const int num, const int channels, const int height, const int width,
    const Dtype* const data, Depthcrafted* grid_features) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / (grid_size * grid_size);
    const int grid_index = index % (grid_size * grid_size);
    const int grid_x = grid_index % grid_size;
    const int grid_y = grid_index / grid_size;
    const int output_idx = (n * grid_size + grid_y) * grid_size + grid_x;
    const int x_start = grid_x * (width / grid_size);
    const int x_end = (grid_x + 1) * (width / grid_size) - 1;
    const int y_start = grid_y * (height / grid_size);
    const int y_end = (grid_y + 1) * (height / grid_size) - 1;
    float depth_mean[3] = {0};
    float depth_histo[3][8] = {0};

    for (int c = 0; c < channels; c++){
      const Dtype* data_slice = data + (n * channels + c) * height * width;
      for (int y = y_start; y < y_end; y++) {
	for (int x = x_start; x < x_end; x++) {
	  depth_mean[c] += data_slice[y * width + x];
	  int bin_idx = (data_slice[y * width + x]+127) / 32;
	  depth_histo[c][bin_idx] += 1;
	}
      }
      depth_mean[c] /= ((width / grid_size) * (height / grid_size));
      for (int j = 0; j < 8; j++) {
        depth_histo[c][j] /= ((width / grid_size) * (height / grid_size));
      }
    }
    Depthcrafted* current_hc = &(grid_features[output_idx]);
    for (int i = 0; i < channels; i++) {
      current_hc->depth_mean[i] = depth_mean[i];
      for (int j = 0; j < 8; j++) {
        current_hc->depth_histogram[i][j] = depth_histo[i][j];
      }
    }
  }
}

// Calculate Region features and label
template <typename Dtype>
__global__ void CalculateRegionFeatures(const int nthreads,
    const int num, const int channels, const int height, const int width, 
    const int R, const int slic_xdim, const int slic_ydim, 
					const int spixel_size, const Dtype* const data, const Dtype* const filldata, const Dtype* const gapdata,
    const Dtype* const slic_index_data, const Dtype* const query_indexes,
    Depthcrafted* query_features) {

  CUDA_KERNEL_LOOP(c_index, nthreads) {
    const int n = c_index / channels / R;
    const int index = c_index / channels;
    const int current_channel = c_index % channels;
    const int query_sp_idx = query_indexes[index];
    const int slic_yind = query_sp_idx / slic_xdim;
    const int slic_xind = query_sp_idx % slic_xdim;
    const int x_start = max((slic_xind - 2) * spixel_size, 0);
    const int x_end = min((slic_xind + 2) * spixel_size, width);
    const int y_start = max((slic_yind - 2) * spixel_size, 0);
    const int y_end = min((slic_yind + 2) * spixel_size, height);
    float depth_mean = 0;
    float depth_histo[8] = {0};

    const Dtype* data_slice = data + (n * channels +current_channel) * height * width;
    int count = 0;
    for (int y = y_start; y < y_end; y++) {
      for (int x = x_start; x < x_end; x++) {
        if (slic_index_data[(n * height + y) * width + x] == query_sp_idx) {
          count += 1;
          depth_mean += data_slice[y * width + x];
	  int bin_idx = (data_slice[y * width + x]+127) / 32;
          depth_histo[bin_idx] += 1;
        }
      }
    }
    if (count == 0) {
      count = 1;
    }
    depth_mean /= count;
    for (int j = 0; j < 8; j++) {
      depth_histo[j] /= count;
    }

    Depthcrafted* current_hc = &(query_features[index]);
    current_hc->depth_mean[current_channel] = depth_mean;
    for (int j = 0; j < 8; j++) {
      current_hc->depth_histogram[current_channel][j] = depth_histo[j];
    }

    // input fill features
    float fill_mean = 0;
    const Dtype* filldata_slice = filldata + (n * channels + current_channel) * height * width;
    count = 0;
    for (int y = y_start; y < y_end; y++) {
      for (int x = x_start; x < x_end; x++) {
        if (slic_index_data[(n * height + y) * width + x] == query_sp_idx) {
          count += 1;
          fill_mean += filldata_slice[y * width + x];
        }
      }
    }
    if (count == 0) {
      count = 1;
    }
    fill_mean /= count;
    current_hc->fill_mean[current_channel] = fill_mean;

    // input gap features
    float gap_mean = 0;
    const Dtype* gapdata_slice = gapdata + (n * channels + current_channel) * height * width;
    count = 0;
    for (int y = y_start; y < y_end; y++) {
      for (int x = x_start; x < x_end; x++) {
        if (slic_index_data[(n * height + y) * width + x] == query_sp_idx) {
          count += 1;
          gap_mean += gapdata_slice[y * width + x];
        }
      }
    }
    if (count == 0) {
      count = 1;
    }
    gap_mean /= count;
    current_hc->gap_mean[current_channel] = gap_mean;
  }
}

__device__ inline int GetOffset(const int n, const int c, const int h,
    const int w, const int C, const int H, const int W) {
  return (((n * C + c) * H + h) * W + w);
}

// Calculate distance between query and grid features
template <typename Dtype>
__global__ void CalculateDistanceBetweenQueryAndGrid(const int nthreads,
    const int N, const int R, const int channels, const int height,
    const int width, const int grid_size,
    const int dim_output, const Depthcrafted* grid_features,
    const Depthcrafted* query_features, Dtype* const top_data) {

  CUDA_KERNEL_LOOP(c_index, nthreads) {
    const int n = c_index / channels / R;
    const int index = c_index / channels;
    const int current_channel = c_index % channels;

    const Depthcrafted* sliced_grid_features =
      &(grid_features[n * grid_size * grid_size]);
    const Depthcrafted* current_query_features = &(query_features[index]);
    
    for (int gx = 0; gx < grid_size; gx++) {
      for (int gy = 0; gy < grid_size; gy++) {
	int chidx = current_channel * (dim_output - 4) / channels;
	float fill_val = current_query_features->fill_mean[current_channel];
	*(top_data + GetOffset(index, chidx, gy, gx, 
			       dim_output, grid_size, grid_size)) = fill_val/256;
	chidx += 1;
	float gap_val = current_query_features->gap_mean[current_channel];
	*(top_data + GetOffset(index, chidx, gy, gx, 
			       dim_output, grid_size, grid_size)) = gap_val/256;
	if (current_channel == channels - 1){
	  int grid_index = gy * grid_size + gx;
	  float grid_val = sliced_grid_features[grid_index].depth_mean[current_channel];
	  float query_val = current_query_features->depth_mean[current_channel];
	  *(top_data + GetOffset(index, dim_output - 4, gy, gx, 
				 dim_output, grid_size, grid_size)) = query_val/256;
	  *(top_data + GetOffset(index, dim_output - 3, gy, gx, 
				 dim_output, grid_size, grid_size)) = grid_val/256;
	  *(top_data + GetOffset(index, dim_output - 2, gy, gx, 
				 dim_output, grid_size, grid_size)) = (query_val - grid_val)/256;
	  float sum = 0;
          for (int b = 0; b < 8; b++) {
            float tmp1 = sliced_grid_features[grid_index].depth_histogram[current_channel][b];
            float tmp2 = current_query_features->depth_histogram[current_channel][b];
            sum += 2 * (tmp1 - tmp2) * (tmp1 - tmp2) / (tmp1 + tmp2 + 0.00000001);
          }
          *(top_data + GetOffset(index, dim_output - 1 , gy, gx,
				 dim_output, grid_size, grid_size)) = sum / 4.0;
	}	  
      }
    }
  }
}

// ---------------------------------------------------
// 
//  Caffe forward implementation
//
// ---------------------------------------------------
template <typename Dtype>
void LowlevelDepthLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(1), 3);
  CHECK_EQ(bottom[1]->shape(1), 3);
  CHECK_EQ(bottom[2]->shape(1), 3);

  int depth_channels = 3;
  int block_cnt = N_ * grid_size_ * grid_size_;
  Depthcrafted* grid_fptr = grid_features_.mutable_gpu_data();
  const Dtype* depth_gpu_ptr = bottom[0]->mutable_gpu_data();
  CalculateGridFeatures<<<CAFFE_GET_BLOCKS(block_cnt),
    CAFFE_CUDA_NUM_THREADS>>>(block_cnt, grid_size_, N_, depth_channels,
        H_, W_, depth_gpu_ptr, grid_fptr);
  
  block_cnt = N_ * R_ * depth_channels;
  int slic_xdim = bottom[4]->shape(3);
  int slic_ydim = bottom[4]->shape(2);

  const Dtype* slic_idx_ptr = bottom[3]->gpu_data();
  const Dtype* query_idx_ptr = bottom[5]->gpu_data();
  Depthcrafted* query_fptr = query_features_.mutable_gpu_data();
  const Dtype* fill_gpu_ptr = bottom[1]->mutable_gpu_data();
  const Dtype* gap_gpu_ptr = bottom[2]->mutable_gpu_data();
  
  CalculateRegionFeatures<<<CAFFE_GET_BLOCKS(block_cnt),
    CAFFE_CUDA_NUM_THREADS>>>(block_cnt, N_, depth_channels, H_, W_, R_,
			      slic_xdim, slic_ydim, sp_size_, depth_gpu_ptr, fill_gpu_ptr,
			      gap_gpu_ptr, slic_idx_ptr, query_idx_ptr, query_fptr);
 
  const Depthcrafted* updated_grid_fptr = grid_features_.gpu_data();
  const Depthcrafted* updated_query_fptr = query_features_.gpu_data();

  CalculateDistanceBetweenQueryAndGrid<<<CAFFE_GET_BLOCKS(block_cnt),
    CAFFE_CUDA_NUM_THREADS>>>(block_cnt, N_, R_, depth_channels, H_, W_, grid_size_, 
        dim_output_, updated_grid_fptr, updated_query_fptr,
			      top[0]->mutable_gpu_data());
}
 INSTANTIATE_LAYER_GPU_FUNCS(LowlevelDepthLayer);
    } // namespace caffe
