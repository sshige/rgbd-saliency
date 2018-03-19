#include "caffe/layers/lowlevel_depth_layer.hpp"

namespace caffe {

template<typename Dtype>
void LowlevelDepthLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  LowlevelDepthParameter ld_param =
    this->layer_param_.lowlevel_depth_param(); 
  grid_size_ = ld_param.grid_size();
  R_ = ld_param.train_region_per_image();
  N_ = bottom[0]->shape(0);
  H_ = bottom[0]->shape(2);
  W_ = bottom[0]->shape(3);
  sp_size_ = ld_param.spixel_size();

  dim_output_ = 10;

  vector<int> fixed_shape(4);
  if (R_ == 0) {
    fixed_shape[0] = N_ * bottom[4]->shape(2) * bottom[4]->shape(3);
    R_ = bottom[4]->shape(2) * bottom[4]->shape(3);
  } else {
    fixed_shape[0] = N_ * R_;
  }
  fixed_shape[1] = dim_output_;
  fixed_shape[2] = grid_size_;
  fixed_shape[3] = grid_size_;
  top[0]->Reshape(fixed_shape);

  fixed_shape[0] = N_;
  fixed_shape[1] = 1;
  grid_features_.Reshape(fixed_shape);

  vector<int> tmp_shape(2);
  tmp_shape[0] = N_;
  tmp_shape[1] = R_;
  query_features_.Reshape(tmp_shape);
  query_region_indexes_.Reshape(tmp_shape);
}

#ifdef CPU_ONLY
STUB_GPU(LowlevelDepthLayer);
#endif

INSTANTIATE_CLASS(LowlevelDepthLayer);
REGISTER_LAYER_CLASS(LowlevelDepth);
}
