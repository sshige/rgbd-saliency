//  lbe.cpp: implementation of the SaliencyLBE class
//------------------------------------------------------------------
//  Copyright (c) 2016 David Feng NICTA/Data61. All rights reserved.
//  Email: firstname.lastname@data61.csiro.au
//------------------------------------------------------------------

#include "lbe.hpp"
#include <numeric>

SaliencyLBE::SaliencyLBE(int n)
: m_n (n)
{}

void SaliencyLBE::computeLBE(const cv::Mat &depth_32FC1, const cv::Mat &seg_32SC1, cv::Mat &sal_32FC1)
{
    m_superpixels = std::unique_ptr<SuperPixels>(new SuperPixels (seg_32SC1, depth_32FC1));
    double r = 0.5*cv::norm(cv::Vec2d(seg_32SC1.rows,seg_32SC1.cols));
    sal_32FC1.create(depth_32FC1.size(), CV_32FC1);
    for (int id=0; id<m_superpixels->size(); id++)
    {
        std::vector<int> neighbours = m_superpixels->getNeighbours(id, r);
        double lbe_score = computeFillScore(id, neighbours) * computeGapScore(id, neighbours);
        sal_32FC1.setTo(lbe_score, m_superpixels->getSuperPixelMask(id));
    }
    cv::normalize(sal_32FC1, sal_32FC1, 1, 0, cv::NORM_INF);
}

void SaliencyLBE::computeFillGap(const cv::Mat &depth_32FC1, const cv::Mat &seg_32SC1, std::vector<std::vector<double> > &fill_list,  std::vector<std::vector<double> > &gap_list, int n_partitions)
{
    m_superpixels = std::unique_ptr<SuperPixels>(new SuperPixels (seg_32SC1, depth_32FC1));
    double r = 0.5*cv::norm(cv::Vec2d(seg_32SC1.rows,seg_32SC1.cols));
    fill_list.clear();
    gap_list.clear();
    for (int id=0; id<m_superpixels->size(); id++)
    {
        std::vector<int> neighbours = m_superpixels->getNeighbours(id, r);
        fill_list.push_back(computeFillScoreList(id, neighbours));
        gap_list.push_back(computeGapScoreList(id, neighbours));
    }
}

std::vector<std::vector<double> > SaliencyLBE::partitionNeighbours(int id, std::vector<int> neighbours, double partition_size)
{
    std::vector<std::vector<double> > neighbourhood_partition(m_n);
    const double max_depth_diff = m_n * partition_size;
    for (int j=0; j<neighbours.size(); j++)
    {
        double depth_diff = m_superpixels->getDepth(neighbours.at(j)) - m_superpixels->getDepth(id);
        // skip if neighbour in front of candidate patch
        if (depth_diff < 0)
        {
            continue;
        }
        int partition;
        // can restructure this conditional to be more clear
        if (depth_diff >= max_depth_diff)
        {
            // avoid overflow issues when partition_size is small
            partition = m_n-1;
        }
        else
        {
            partition = std::min(m_n-1, static_cast<int>(depth_diff / partition_size));
        }
        double theta = getAngle(m_superpixels->getXY(id), m_superpixels->getXY(neighbours.at(j)));
        neighbourhood_partition.at(partition).push_back(theta);
    }
    return neighbourhood_partition;
}

// use atan2
double SaliencyLBE::getAngle(const cv::Point2d &a, const cv::Point2d &b)
{
    double theta;
    double dy = b.y - a.y;
    double dx = b.x - a.x;
    if (dx==0)
    {
        if (dy==0) return NAN;
        if (dy>0) return 0;
        else return CV_PI;
    }
    else if (dy==0)
    {
        if (dx>0) return 0.5*CV_PI;
        else return 1.5*CV_PI;
    }
    else
    {
        theta = std::atan(dx/dy);
        if (dx>0 && dy<0) theta = CV_PI + theta;
        else if (dx<0 && dy<0) theta = CV_PI + theta;
        else if (dx<0 && dy>0) theta = 2*CV_PI + theta;
        return theta;
    }
}

double SaliencyLBE::getGapScore(std::vector<double> &angles)
{
    if (angles.size() < 2)
    {
        return 0.0;
    }
    double max_gap = 0;
    std::sort(angles.begin(), angles.end());
    for (int i=0; i<angles.size()-1; i++)
    {
        max_gap = std::max(max_gap, angles.at(i+1)-angles.at(i));
    }
    // wrap around angular region
    max_gap = std::max(max_gap, 2*CV_PI-angles.back()+angles.front());
    return 1.0 - 0.5 * max_gap / CV_PI;
}

std::vector<double> SaliencyLBE::computeFillScoreList(int id, std::vector<int> neighbours)
{
    PolarOccurenceHistogram hist(32);
    std::vector<std::vector<double> > neighbourhood_partition = partitionNeighbours(id, neighbours, m_superpixels->getDepthSD(id)/m_n);
    // compute angular fill statistic for increasingly large background sets
    std::vector<double> fill_score_list;
    for (int i=neighbourhood_partition.size()-1; i>=0; i--)
    {
        for (int j=0; j<neighbourhood_partition.at(i).size(); j++)
        {
            hist.add(neighbourhood_partition.at(i).at(j));
        }
        fill_score_list.push_back(hist.getFillRatio());
    }
    return fill_score_list;
}

double SaliencyLBE::computeFillScore(int id, std::vector<int> neighbours)
{
    PolarOccurenceHistogram hist(32);
    std::vector<std::vector<double> > neighbourhood_partition = partitionNeighbours(id, neighbours, m_superpixels->getDepthSD(id)/m_n);
    // compute angular fill statistic for increasingly large background sets
    double fill_score = 0.0;
    for (int i=neighbourhood_partition.size()-1; i>=0; i--)
    {
        for (int j=0; j<neighbourhood_partition.at(i).size(); j++)
        {
            hist.add(neighbourhood_partition.at(i).at(j));
        }
        fill_score += hist.getFillRatio();
    }
    return fill_score / m_n;
}

double SaliencyLBE::computeGapScore(int id, std::vector<int> neighbours)
{
    PolarOccurenceHistogram hist(32);
    std::vector<std::vector<double> > neighbourhood_partition = partitionNeighbours(id, neighbours, m_superpixels->getDepthSD()/m_n);
    // compute angular gap statistic for increasingly large background sets
    double gap_score = 0.0;
    std::vector<double> background_set;
    for (int i=neighbourhood_partition.size()-1; i>=0; i--)
    {
        background_set.insert(background_set.end(),
                              neighbourhood_partition.at(i).begin(),
                              neighbourhood_partition.at(i).end());
        gap_score += getGapScore(background_set);
    }
    return gap_score / m_n;
}

std::vector<double> SaliencyLBE::computeGapScoreList(int id, std::vector<int> neighbours)
{
    PolarOccurenceHistogram hist(32);
    std::vector<std::vector<double> > neighbourhood_partition = partitionNeighbours(id, neighbours, m_superpixels->getDepthSD()/m_n);
    // compute angular gap statistic for increasingly large background sets
    std::vector<double> gap_score_list, background_set;
    for (int i=neighbourhood_partition.size()-1; i>=0; i--)
    {
        background_set.insert(background_set.end(),
                              neighbourhood_partition.at(i).begin(),
                              neighbourhood_partition.at(i).end());
        gap_score_list.push_back(getGapScore(background_set));
    }
    return gap_score_list;
}


SaliencyLBE::PolarOccurenceHistogram::PolarOccurenceHistogram(int nbins)
: m_nbins (nbins)
, m_bin_step (2*CV_PI / static_cast<double>(nbins))
, m_data (std::vector<bool>(nbins))
{ }

void SaliencyLBE::PolarOccurenceHistogram::add(double theta)
{
    m_data.at(static_cast<int>(theta / m_bin_step) % m_nbins) = true;
}

double SaliencyLBE::PolarOccurenceHistogram::getFillRatio()
{
    return std::accumulate(m_data.begin(), m_data.end(), 0) / static_cast<double>(m_nbins);
}









