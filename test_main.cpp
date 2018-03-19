#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/net.hpp>
#include <caffe/util/io.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <boost/filesystem.hpp>
#include <boost/chrono.hpp>
#include <ORUtils/gSLICr_defines.h>

namespace fs = boost::filesystem;
using namespace std;
using namespace cv;
using namespace caffe;
using namespace gSLICr;

float MEAN_BGR[3] = {104, 117, 123};

void CustomResize(const Mat& img, Mat& output, cv::Size size) {

  Mat newmap(size, CV_8UC1);
  float mul_x = img.cols / (float) (size.width);
  float mul_y = img.rows / (float) (size.height);
  for (int i = 0; i < size.height; i++) {
    for (int j = 0; j < size.width; j ++) {
      newmap.at<uchar>(i, j) = static_cast<uchar>(img.at<float>((int)(i*mul_y),
          (int)(j*mul_x)) * 255);
    }
  }
  output = newmap.clone();
}

void CustomCVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;

  for (int ch = 0; ch < datum_channels; ch++) {
    for (int r = 0; r < datum_height; r++) {
      for (int c = 0; c < datum_width; c++) {
	unsigned char mat_val = cv_img.at<Vec3b>(r, c)[ch];
	datum->add_float_data(static_cast<float>(mat_val) - MEAN_BGR[ch]);
      }
    }
  }
}

void CustomDepthCVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;

  for (int ch = 0; ch < datum_channels; ch++) {
    for (int r = 0; r < datum_height; r++) {
      for (int c = 0; c < datum_width; c++) {
      unsigned char mat_val = cv_img.at<Vec3b>(r, c)[ch];
      datum->add_float_data(static_cast<float>(mat_val) - 127);
      }
    }
  }
}

int main(int argc, char** argv) {
  const int fixed_size = 324;
  string net_proto_path(argv[1]);
  string net_binary_path(argv[2]);
  string rgb_dirpath = argv[3];
  string depth_dirpath = argv[4];
  string fill_dirpath = argv[5];
  string gap_dirpath = argv[6];
  string save_dirpath = argv[7];
  double total_count = 0;

  std::unique_ptr<Net<float>> caffe_test_net;
  Caffe::set_mode(Caffe::Brew::GPU);
  Caffe::SetDevice(0);
  caffe_test_net.reset(new Net<float>(net_proto_path, caffe::TEST));
  caffe_test_net->CopyTrainedLayersFrom(net_binary_path);

  // collect filenames
  fs::directory_iterator rgbdir_iter = fs::directory_iterator(rgb_dirpath);
  fs::directory_iterator rgb_end_iter;
  vector<string> rgb_filenames;
  while (rgbdir_iter != rgb_end_iter) {
    string rgbname = rgbdir_iter->path().string();
    rgb_filenames.push_back(rgbname);
    rgbdir_iter++;
  }
  std::sort(rgb_filenames.begin(), rgb_filenames.end());

  fs::directory_iterator depthdir_iter = fs::directory_iterator(depth_dirpath);
  fs::directory_iterator depth_end_iter;
  vector<string> depth_filenames;
  while (depthdir_iter != depth_end_iter) {
    string depthname = depthdir_iter->path().string();
    depth_filenames.push_back(depthname);
    depthdir_iter++;
  }
  std::sort(depth_filenames.begin(), depth_filenames.end());

  fs::directory_iterator filldir_iter = fs::directory_iterator(fill_dirpath);
  fs::directory_iterator fill_end_iter;
  vector<string> fill_filenames;
  while (filldir_iter != fill_end_iter) {
    string fillname = filldir_iter->path().string();
    fill_filenames.push_back(fillname);
    filldir_iter++;
  }
  std::sort(fill_filenames.begin(), fill_filenames.end());

  fs::directory_iterator gapdir_iter = fs::directory_iterator(gap_dirpath);
  fs::directory_iterator gap_end_iter;
  vector<string> gap_filenames;
  while (gapdir_iter != gap_end_iter) {
    string gapname = gapdir_iter->path().string();
    gap_filenames.push_back(gapname);
    gapdir_iter++;
  }
  std::sort(gap_filenames.begin(), gap_filenames.end());

  if (rgb_filenames.size() != depth_filenames.size()){
    std::cout << "error image num in loading image" << endl;
    return 0;
  }
  if (rgb_filenames.size() != fill_filenames.size()){
    std::cout << "error image num in loading image" << endl;
    return 0;
  }
  if (rgb_filenames.size() != gap_filenames.size()){
    std::cout << "error image num in loading image" << endl;
    return 0;
  }
  int total_num = rgb_filenames.size();

  for (int imgidx = 0; imgidx < total_num; imgidx++) {
    string rgb_path = rgb_filenames[imgidx];
    string depth_path = depth_filenames[imgidx];
    string fill_path = fill_filenames[imgidx];
    string gap_path = gap_filenames[imgidx];
    Mat rgb_image = cv::imread(rgb_path);
    Mat depth_image = cv::imread(depth_path);
    Mat fill_image = cv::imread(fill_path);
    Mat gap_image = cv::imread(gap_path);
    boost::chrono::system_clock::time_point start;
    start = boost::chrono::system_clock::now();
    fs::path savepath(save_dirpath);
    fs::path rgb_path_p(rgb_path);
    fs::path depth_path_p(depth_path);
    fs::path fill_path_p(fill_path);
    fs::path gap_path_p(gap_path);

    cv::Size original_size = cv::Size(rgb_image.cols, rgb_image.rows);
    cv::resize(rgb_image, rgb_image, cv::Size(fixed_size, fixed_size));
    cv::resize(depth_image, depth_image, cv::Size(fixed_size, fixed_size));
    cv::resize(fill_image, fill_image, cv::Size(fixed_size, fixed_size));
    cv::resize(gap_image, gap_image, cv::Size(fixed_size, fixed_size));

    vector<Datum> input_rgb_datum;
    input_rgb_datum.emplace_back();
    CustomCVMatToDatum(rgb_image, &(input_rgb_datum[0]));
    boost::shared_ptr<MemoryDataLayer<float>> input_rgb_layer =
      boost::static_pointer_cast<MemoryDataLayer<float>>(
							 caffe_test_net->layer_by_name("data"));
    input_rgb_layer->AddDatumVector(input_rgb_datum);

    vector<Datum> input_depth_datum;
    input_depth_datum.emplace_back();
    CustomDepthCVMatToDatum(depth_image, &(input_depth_datum[0]));
    boost::shared_ptr<MemoryDataLayer<float>> input_depth_layer =
      boost::static_pointer_cast<MemoryDataLayer<float>>(
							 caffe_test_net->layer_by_name("depthdata"));
    input_depth_layer->AddDatumVector(input_depth_datum);

    vector<Datum> input_fill_datum;
    input_fill_datum.emplace_back();
    CustomDepthCVMatToDatum(fill_image, &(input_fill_datum[0]));
    boost::shared_ptr<MemoryDataLayer<float>> input_fill_layer =
      boost::static_pointer_cast<MemoryDataLayer<float>>(
							 caffe_test_net->layer_by_name("filldata"));
    input_fill_layer->AddDatumVector(input_fill_datum);

    vector<Datum> input_gap_datum;
    input_gap_datum.emplace_back();
    CustomDepthCVMatToDatum(gap_image, &(input_gap_datum[0]));
    boost::shared_ptr<MemoryDataLayer<float>> input_gap_layer =
      boost::static_pointer_cast<MemoryDataLayer<float>>(
							 caffe_test_net->layer_by_name("gapdata"));
    input_gap_layer->AddDatumVector(input_gap_datum);

    vector<Blob<float>*> empty_vec;
    caffe_test_net->Forward(empty_vec);

    const boost::shared_ptr<Blob<float>> slic_blob =
      caffe_test_net->blob_by_name("slic");
    const boost::shared_ptr<Blob<float>> score_blob =
      caffe_test_net->blob_by_name("score");
    const float* score_ptr = score_blob->cpu_data();
    Mat result(fixed_size, fixed_size, CV_8UC1);

    const float* slic_ptr = slic_blob->cpu_data();
    float score_sum = 0;
    int score_count = 0;
    for (int i = 0; i < fixed_size; i++) {
      for (int j = 0; j < fixed_size; j++) {
	const float index = slic_ptr[i*fixed_size + j];
	const float score = score_ptr[(int)(index)];
	result.at<uchar>(i, j) = static_cast<uchar>(score*255);
      }
    }

    resize(result, result, original_size);
    boost::chrono::duration<double> sec2 = boost::chrono::system_clock::now() - start;
    cout << "time : " << sec2.count() << "s" << endl;
    total_count += (double)sec2.count();

    string outimg = rgb_path_p.filename().string();
    outimg.replace(outimg.end()-4, outimg.end(), ".png");
    fs::path filename(outimg);
    fs::path outputpath = savepath / outimg;
    imwrite(outputpath.string(), result);
    cout << outputpath << endl;
  }

  cout <<  "AVG time : " << total_count / total_num << "s" << endl;
  return 0;
}
