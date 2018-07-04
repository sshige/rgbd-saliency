//  superpixels.hpp: implementation of the SuperPixels interface
//------------------------------------------------------------------
//  This code implements the SuperPixels helper class for
//  SaliencyLBE. This class computes and stores superpixel
//  properties such as mean depth and superpixel neighbourhoods.
//
//  Note that this class does not perform superpixel segmentation.
//  The segmentation is supplied as an argument to the constructor.
//------------------------------------------------------------------
//  Copyright (c) 2016 David Feng NICTA/Data61. All rights reserved.
//  Email: firstname.lastname@data61.csiro.au
//------------------------------------------------------------------

#ifndef __LBE__superpixels__
#define __LBE__superpixels__

#include <stdio.h>
#include <opencv2/opencv.hpp>

// Computes patch properties from depth image and superpixel segmentation.
class SuperPixels
{
public:
    // Stores properties for each superpixel
    struct SuperPixel
    {
    public:
        cv::Point2d m_centroid;
        double m_depth;
        double m_local_depth_sd;
        std::vector<std::pair<int,double> > m_neighbours;
    };
    
    // Computes superpixel properties from depth and superpixel segmentation
    // images.
    //
    // The input depth image is 32 bit float format.
    //
    // The input superpixel segmentation image seg_32SC1 is 32 bit integer
    // format storing the superpixel ids of each pixel. Superpixel ids should
    // be consecutive integers starting from zero.
    SuperPixels(const cv::Mat &seg_32SC1, const cv::Mat &depth_32FC1);
    
    // generate binary superpixel mask image
    cv::Mat getSuperPixelMask(int id);
    
    // number of superpixels
    size_t size();
    
    // get mean depth of selected superpixel
    double getDepth(int id);
    
    // get superpixel centroid
    cv::Point2d getXY(int id);
    
    // list of labels for superpixels with centroid within radius r from
    //  centroid of selected superpixel
    std::vector<int> getNeighbours(int id, double r);
    
    // Get the standard deviation of mean depths of patches in local
    // neighbourhood of sperpixel with given id.
    //
    // If input id is -1 then return the standard deviation of mean depths of
    // the entire image.
    double getDepthSD(int id=-1);
    
private:
    // compute standard deviation of mean depths of superpixels
    double computeDepthSD(const std::vector<int> &patches);
    
    // initialise centroids from segmentation image
    void superPixelsFromSeg(const cv::Mat &seg_32SC1);
    
    // compute mean depth and depth standard deviation
    void addDepth(const cv::Mat &depth_32FC1);
    
    // compute sorted pairwise distances between superpixels
    void computeNeighbourDistances();
    
    std::vector<SuperPixel> m_superpixels;
    cv::Mat m_seg_32SC1;
    double m_depth_sd;
};

#endif /* defined(__LBE__superpixels__) */
