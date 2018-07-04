//  main.cpp: demo of LBE saliency
//------------------------------------------------------------------
//  This code computes the LBE saliency maps given a set of depth
//  and RGB image pairs. Note that the RGB images are only required
//  for computing the superpixel segmentation.
//
//  The demo takes four command line arguments:
//    1. path to folder containing depth images
//    2. path to folder containing rgb images
//    3. path to output folder where computed lbe map will
//       be saved
//    4. (optional) image extension of depth and RGB files. Default
//       value is .jpg
//------------------------------------------------------------------
//  Copyright (c) 2016 David Feng NICTA/Data61. All rights reserved.
//  Email: firstname.lastname@data61.csiro.au
//------------------------------------------------------------------

#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "SLIC.h"
#include "lbe.hpp"
#include <dirent.h>
#include <sys/stat.h>

/*******************************************************************************
 * Calls SLIC segmentation code from
 *  "SLIC superpixels compared to state-of-the-art superpixel methods"
 *  Achanta et al. 2012
 ******************************************************************************/

// Performs SLIC superpixel segmentation of input RGB image.
// Returns 32-bit integer image of superpixel labels.
cv::Mat getSLICOSegmentation(cv::Mat &rgb_8UC3)
{
  cv::Mat seg_32SC1 (rgb_8UC3.size(), CV_32SC1);
  SLIC slic;
  int numPixels = rgb_8UC3.rows*rgb_8UC3.cols;
  seg_32SC1.create(rgb_8UC3.size(), CV_32SC1);
  int numSegments = int(sqrt((double)rgb_8UC3.rows*rgb_8UC3.rows+rgb_8UC3.cols*rgb_8UC3.cols));
  int numlabels(0);
  int* labels = (int *)seg_32SC1.data;
  slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(rgb_8UC3.data, rgb_8UC3.cols, rgb_8UC3.rows, labels, numlabels, numSegments, 18.0); // original 20
  return seg_32SC1;
}


/*******************************************************************************
 * File Operations
 ******************************************************************************/

// return a list of files in directory with given extension
std::vector<std::string> getExtFilesInDir ( std::string dir_path, std::string ext )
{
  std::vector<std::string> out;
  DIR *dir;
  struct dirent *ent;
  if( ( dir = opendir( dir_path.c_str() ) ) != NULL )
    {
      while( ( ent = readdir (dir) ) != NULL )
        {
          std::string file_name (ent->d_name);
          size_t f_len = file_name.size();
          size_t i_len = ext.size();
          if ( f_len >= i_len && file_name.substr( f_len - i_len, i_len ).compare( ext ) == 0 )
            {
              out.push_back( file_name );
            }
        }
      closedir( dir );
    }
  return out;
}

// Load disparity image, return as 32-bit float image of normalised depth values.
cv::Mat loadDisparity(std::string image_path)
{
  cv::Mat depth = cv::imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
  depth.convertTo(depth, CV_32FC1);
  cv::normalize(depth, depth, 1, 0, cv::NORM_INF);
  depth = 1 - depth;
  return depth;
}

// Load depth image, return as 32-bit float image of normalised depth values.
cv::Mat loadDepth(std::string image_path)
{
  cv::Mat depth = cv::imread( image_path, CV_LOAD_IMAGE_UNCHANGED );
  depth.convertTo(depth, CV_32FC1, 0.001);
  cv::normalize(depth, depth, 1, 0, cv::NORM_INF);
  return depth;
}

/*******************************************************************************
 * Compute LBE map
 ******************************************************************************/

// Compute LBE saliency maps example images.
void runDemo(std::string depth_dir, std::string rgb_dir, std::string output_dir, std::string ext)
{
  // get list of jpeg filenames in depth directory (assume same names for rgb)
  std::vector<std::string> im_files = getExtFilesInDir(depth_dir, ext);
    
  // create output directory
  mkdir(output_dir.c_str(), 0777);
    
  // lbe computation object
  SaliencyLBE lbe;
    
  for (int i=0; i<im_files.size(); i++)
    {
      std::cout << "[LBE Demo]: " << i+1 << " of " << im_files.size() << std::endl;
      std::cout << "- depth file: " << depth_dir+"/"+im_files.at(i) << std::endl;
        
      // use this to load 8-bit disparity image (NJUDS2000)
      cv::Mat depth_32FC1 = loadDisparity(depth_dir+"/"+im_files.at(i));
        
      // use this to load 16-bit depth image (Kinect output format)
      // cv::Mat depth_32FC1 = loadDepth(depth_dir+"/"+im_files.at(i));
        
      // load corresponding rgb file (used to compute slic segmentaiton)
      std::cout << "- rgb file:   " << rgb_dir+"/"+im_files.at(i) << std::endl;
      cv::Mat rgb_8UC3 = cv::imread(rgb_dir+"/"+im_files.at(i));
        
      // compute segmentation
      cv::Mat seg_32SC1 = getSLICOSegmentation(rgb_8UC3);
        
      // compute and save lbe image
      cv::Mat sal_32FC1;
      lbe.computeLBE(depth_32FC1, seg_32SC1, sal_32FC1);
      sal_32FC1.convertTo(sal_32FC1, CV_8UC1, 255);
      cv::imwrite(output_dir+"/"+im_files.at(i), sal_32FC1);
      std::cout << "- saliency saved to " << output_dir+"/"+im_files.at(i) << std::endl;
    }
}

std::vector<cv::Mat> fillListToImList(std::vector<std::vector<double> > fill_list, cv::Mat seg_32SC1)
{
  std::vector<cv::Mat> im_list, merged_list;
  int n_partitions = fill_list.at(0).size();
  for (int i=0; i<n_partitions; i++)
    {
      cv::Mat tmp (seg_32SC1.size(), CV_8UC1);
      for (int id=0; id<fill_list.size(); id++)
        {
          tmp.setTo(255*fill_list.at(id).at(i), seg_32SC1==id);
        }
      im_list.push_back(tmp);
    }
  return im_list;
}

std::vector<cv::Mat> fillListToImageList(std::vector<std::vector<double> > fill_list, cv::Mat seg_32SC1)
{
  std::vector<cv::Mat> im_list, merged_list;
  int n_partitions = fill_list.at(0).size();
  for (int i=0; i<n_partitions; i++)
    {
      cv::Mat tmp (seg_32SC1.size(), CV_8UC1);
      for (int id=0; id<fill_list.size(); id++)
        {
          tmp.setTo(255*fill_list.at(id).at(i), seg_32SC1==id);
        }
      im_list.push_back(tmp);
    }
    
    
  int start_ind = 0, n_channels = std::min(3, int(fill_list.size()));
  while(start_ind + 3 <= n_partitions)
    {
      merged_list.push_back(cv::Mat (seg_32SC1.size(), CV_8UC3));
      start_ind += 3;
    }
  if (start_ind < n_partitions)
    {
      int n_channels = n_partitions - start_ind;
      int type = -1;;
      switch (n_channels)
        {
        case 1:
          type = CV_8UC3;
          break;
        case 2:
          type = CV_8UC3;
          break;
        }
      merged_list.push_back(cv::Mat::zeros (seg_32SC1.size(), type));
    }
  std::vector<int> from_to;
  for (int i=0; i<n_partitions; i++)
    {
      from_to.push_back(i);
      from_to.push_back(i);
    }
  cv::mixChannels(im_list, merged_list, from_to);
  return merged_list;
}

/*******************************************************************************
 * Compute LBE map
 ******************************************************************************/

// Compute LBE saliency maps example images.
void generateFillGapImages(std::string depth_dir, std::string rgb_dir, std::string fill_dir, std::string gap_dir, std::string ext, int bit_cnt)
{
  // get list of jpeg filenames in depth directory (assume same names for rgb)
  std::vector<std::string> im_files = getExtFilesInDir(depth_dir, ext);

  // create output directory
  mkdir(gap_dir.c_str(), 0777);
  mkdir(fill_dir.c_str(), 0777);

  int n_partitions = 3;

  int n_out = std::ceil(n_partitions / 3.0);

  // lbe computation object
  SaliencyLBE lbe(n_partitions);

  for (int i=0; i<im_files.size(); i++)
    {
      std::cout << "[LBE Demo]: " << i+1 << " of " << im_files.size() << std::endl;
      std::cout << "- depth file: " << depth_dir+"/"+im_files.at(i) << std::endl;

      cv::Mat depth_32FC1;

      if (bit_cnt == 8) {
        // use this to load 8-bit disparity image (NJUDS2000)
        depth_32FC1 = loadDisparity(depth_dir+"/"+im_files.at(i));
      }
      else if (bit_cnt == 16) {
        // use this to load 16-bit depth image (Kinect output format)
        depth_32FC1 = loadDepth(depth_dir+"/"+im_files.at(i));
      } else {
        std::cout << "Please specify bit as 8-bit or 16-bit" << std::cout;
        return;
      }

      // load corresponding rgb file (used to compute slic segmentaiton)
      std::cout << "- rgb file:   " << rgb_dir+"/"+im_files.at(i) << std::endl;
      cv::Mat rgb_8UC3 = cv::imread(rgb_dir+"/"+im_files.at(i));

      // compute segmentation
      cv::Mat seg_32SC1 = getSLICOSegmentation(rgb_8UC3);

      // compute and save lbe image
      cv::Mat fill_32FC3, gap_32FC3;
      std::vector<std::vector<double> > fill_list;
      std::vector<std::vector<double> > gap_list;
      lbe.computeFillGap(depth_32FC1, seg_32SC1, fill_list, gap_list, n_partitions);
      std::vector<cv::Mat> fill_im_list = fillListToImageList(fill_list, seg_32SC1);
      std::vector<cv::Mat> gap_im_list = fillListToImageList(gap_list, seg_32SC1);

      for (int j=0; j<n_out; j++)
        {
          cv::imwrite(fill_dir+"/"+im_files.at(i), fill_im_list.at(j));
          cv::imwrite(gap_dir+"/"+im_files.at(i), gap_im_list.at(j));
        }
    }
}

int main(int argc, const char * argv[])
{
  if (argc<6)
    {
      std::cout << "Usage: ./" << argv[0] <<
        " [depth_path] [rgb_path] [output_fill_path] [output_gap_path] [img_ext(optional)] [bit]" << std::endl;
    }

  std::string depth_path = argv[1];
  std::string rgb_path = argv[2];
  std::string fill_path = argv[3];
  std::string gap_path = argv[4];
  std::string ext = argv[5];
  int bit_cnt = std::atoi(argv[6]);

  generateFillGapImages(depth_path, rgb_path, fill_path, gap_path, ext, bit_cnt);

  return 0;
}
