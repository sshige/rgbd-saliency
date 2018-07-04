//  superpixels.cpp: implementation of the SuperPixels class
//------------------------------------------------------------------
//  Copyright (c) 2016 David Feng NICTA/Data61. All rights reserved.
//  Email: firstname.lastname@data61.csiro.au
//------------------------------------------------------------------

#include "superpixels.hpp"
#include <numeric>

SuperPixels::SuperPixels(const cv::Mat &seg_32SC1, const cv::Mat &depth_32FC1)
{
    superPixelsFromSeg(seg_32SC1);
    computeNeighbourDistances();
    addDepth(depth_32FC1);
}

cv::Mat SuperPixels::getSuperPixelMask(int id)
{
    return m_seg_32SC1 == id;
}

std::vector<int> SuperPixels::getNeighbours(int id, double r)
{
    std::vector<int> out;
    
    // get closest superpixels within distance r to id
    for ( int i=0; i<m_superpixels.at(id).m_neighbours.size(); i++ )
    {
        int other = m_superpixels.at(id).m_neighbours.at(i).first;
        double dist = cv::norm(m_superpixels.at(other).m_centroid - m_superpixels.at(id).m_centroid);
        if (dist > r) break;
        out.push_back(other);
    }
    
    return out;
}

void SuperPixels::superPixelsFromSeg(const cv::Mat &seg_32SC1)
{
    // save local copy of segmentation image
    seg_32SC1.copyTo(m_seg_32SC1);
    
    // get superpixel label range [0,k]
    double k;
    cv::minMaxLoc( m_seg_32SC1, NULL, &k );
    
    // compute properties for each superpixel
    for ( int i=0; i<=k; i++ )
    {
        SuperPixel sp;
        
        // get binary mask image for superpixel
        cv::Mat mask_8UC1 = getSuperPixelMask( i );
        
        // centroid (x,y) of superpixel
        cv::Moments m = cv::moments( mask_8UC1, true );
        sp.m_centroid = cv::Point2d( m.m10/m.m00, m.m01/m.m00 );
        
        m_superpixels.push_back( sp );
    }
}

size_t SuperPixels::size()
{
    return m_superpixels.size();
}

void SuperPixels::addDepth(const cv::Mat &depth_32FC1)
{
    for (int i=0; i<m_superpixels.size(); i++)
    {
        m_superpixels.at(i).m_depth = cv::mean(depth_32FC1,getSuperPixelMask(i))[0];
    }
    // compute depth standard deviations of local neighbourhood of each patch
    double r = 0.1*cv::norm(cv::Vec2d(depth_32FC1.rows,depth_32FC1.cols));
    for (int i=0; i<m_superpixels.size(); i++)
    {
        std::vector<int> local_neighbourhood = getNeighbours(i, r);
        m_superpixels.at(i).m_local_depth_sd = computeDepthSD(local_neighbourhood);
    }
    // compute global depth standard deviations
    std::vector<int> ids (size());
    std::iota(ids.begin(), ids.end(), 0);
    m_depth_sd = computeDepthSD(ids);
}

double SuperPixels::getDepthSD(int id)
{
    if (id<0)
    {
        return m_depth_sd;
    }
    else
    {
        return m_superpixels.at(id).m_local_depth_sd;
    }
}

double SuperPixels::getDepth(int id)
{
    return m_superpixels.at(id).m_depth;
}

cv::Point2d SuperPixels::getXY(int id)
{
    return m_superpixels.at(id).m_centroid;
}

double SuperPixels::computeDepthSD (const std::vector<int> &patches)
{
    if (patches.size()==0) return -1;
    double var = 0;
    double mean_depth = 0;
    for (int i=0; i<patches.size(); i++)
    {
        mean_depth += getDepth(patches.at(i));
    }
    mean_depth /= patches.size();
    for (int j=0; j<patches.size(); j++)
    {
        double depth_diff = mean_depth - getDepth(patches.at(j));
        var += depth_diff * depth_diff;
    }
    return std::sqrt(var / patches.size());
}

void SuperPixels::computeNeighbourDistances()
{
    ////////////////////////////////////////////////////////////////////
    // Set up superpixel distance table for finding neighbours
    ////////////////////////////////////////////////////////////////////
    
    // get sorted pairwise superpixel distance
    for ( int i=0; i<m_superpixels.size(); i++ )
    {
        m_superpixels.at(i).m_neighbours.clear();
        cv::Point2d ci = m_superpixels.at( i ).m_centroid;
        // get list of neighbour distances
        // can halve number of distance computations
        for ( int j = 0; j < m_superpixels.size(); j++ )
        {
            if ( i == j )
            {
                continue;
            }
            m_superpixels.at(i).m_neighbours.push_back(std::pair<int,double>(j, cv::norm(ci - m_superpixels.at(j).m_centroid)));
        }
        // sort neighbours by distance
        std::sort(m_superpixels.at(i).m_neighbours.begin(),
                  m_superpixels.at(i).m_neighbours.end(),
                  []( const std::pair<int,double> &left,
                     const std::pair<int,double> &right )
                  { return left.second < right.second; } );
    }

}