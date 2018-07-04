//  lbe.hpp: implementation of the SaliencyLBE interface
//------------------------------------------------------------------
//  This code implements the LBE depth saliency method described in:
//
//  David Feng, Nick Barnes, Shaodi You, Chris McCarthy
//  "Local Background Enclosure for RGB-D Salient Object Detection"
//  CVPR, June 2016
//------------------------------------------------------------------
//  Copyright (c) 2016 David Feng NICTA/Data61. All rights reserved.
//  Email: firstname.lastname@data61.csiro.au
//------------------------------------------------------------------

#ifndef __LBE__lbe__
#define __LBE__lbe__

#include <stdio.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include "superpixels.hpp"

// computes saliency map by measuring local background enclosure
// Example:
//     SaliencyLBE() lbe;
//     cv::Mat saliency;
//     lbe.computeLBE(depth_image, segmenatation_image, saliency);
class SaliencyLBE
{
public:
    
    // n is the number of integral discretisation steps when computing
    // distribution functions
    SaliencyLBE(int n = 10);
    
    // Generate LBE saliency map by computing the distribution functions of
    // angular fill and gap and returning their product.
    //
    // The input depth image depth_32FC1 is a 32-bit float format.
    //
    // The input superpixel segmentation image seg_32SC1 is 32 bit integer
    // format storing the superpixel ids of each pixel. Superpixel ids are
    // sequential starting from zero.
    //
    // The output saliency sal_32FC1 will be in 32-bit float format with
    // normalised values in [0,1], and does not need to be initialised.
    void computeLBE(const cv::Mat &depth_32FC1, const cv::Mat &seg_32SC1, cv::Mat &sal_32FC1);
    
    void computeFillGap(const cv::Mat &depth_32FC1, const cv::Mat &seg_32SC1, std::vector<std::vector<double> > &fill_list, std::vector<std::vector<double> > &gap_list, int n_partitions);
    
private:
    // Boolean polar histogram, stores true/false value for each bin.
    //
    // This class is used by SaliencyLBE to compute the angular fill statistic
    class PolarOccurenceHistogram
    {
    public:
        // nbins is the number of bins in the histogram
        PolarOccurenceHistogram(int nbins);
        // sets bin corresponding to theta as true
        void add(double theta);
        // return proportion of sectors with value true
        double getFillRatio();
    private:
        std::vector<bool> m_data;
        const int m_nbins;
        // angular size in radians of each bin
        const double m_bin_step;
    };
    
    // angular fill distribution function
    double computeFillScore(int id, std::vector<int> neighbours);
    std::vector<double> computeFillScoreList(int id, std::vector<int> neighbours);
    // angular gap distribution function
    double computeGapScore(int id, std::vector<int> neighbours);
    std::vector<double> computeGapScoreList(int id, std::vector<int> neighbours);
    // partition neighbours based on depth difference with candidate patch
    std::vector<std::vector<double> > partitionNeighbours(int id, std::vector<int> neighbours, double partition_size);
    // compute depth standard deviation of patches
    double computeDepthSD (const std::vector<int> &patches);
    // get between line a->b and line a->(a+(0,1))
    double getAngle(const cv::Point2d &a, const cv::Point2d &b);
    // get size of largest gap between input angles
    double getGapScore(std::vector<double> &angles);
    
    // the number of integral discretisation steps when computing
    // distribution functions
    int m_n;
    // superpixels object which stores centroid, depth, and neighbor information
    // for each superpixel
    std::unique_ptr<SuperPixels> m_superpixels;
};

#endif /* defined(__LBE__lbe__) */
