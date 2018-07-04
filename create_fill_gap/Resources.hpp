//
//  Resources.h
//  RGBDSaliency
//
//  Created by David Feng on 9/11/2014.
//  Copyright (c) 2014 DavidFeng. All rights reserved.
//

#ifndef __RGBDSaliency__Resources__
#define __RGBDSaliency__Resources__

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <sys/stat.h>
#include <iostream>
#include <fstream>

namespace ex
{
    class Resources {
    public:
        Resources( std::string dir_file_path )
        : m_image_ext( std::vector<std::string>{ ".png", ".jpg", ".bmp", ".PNG", ".JPG", ".BMP" } )
        {
            loadDirectoryList( dir_file_path );
        }
        void loadDirectoryList( std::string dir_file_path )
        {
            cv::FileStorage fs ( dir_file_path, cv::FileStorage::READ );
            fs["depth_dir"] >> m_depth_dir;
            fs["depth_raw_dir"] >> m_depth_raw_dir;
            fs["rgb_dir"] >> m_rgb_dir;
            fs["ground_truth_dir"] >> m_ground_truth_dir;
            fs["ground_truth_boundaries_dir"] >> m_ground_truth_boundaries_dir;
            fs["segmentation_dir"] >> m_segmentation_dir;
            fs["seg_dir"] >> m_seg_dir;
            fs["normals_dir"] >> m_normals_dir;
            fs["output_dir"] >> m_output_dir;
            fs["results_dir"] >> m_results_dir;
            fs["image_dir"] >> m_image_dir;
            fs["peng_dir"] >> m_peng_dir;
            fs["bp_dir"] >> m_background_prior_dir;
            fs["dp_dir"] >> m_depth_prior_dir;
            fs["op_dir"] >> m_orientation_prior_dir;
            fs["rc_dir"] >> m_region_contrast_dir;
            fs["fgs_dir"] >> m_fill_graph_sd_dir;
            fs["ggs_dir"] >> m_gap_graph_sd_dir;
            fs["cc_dir"] >> m_colour_contrast_dir;
            fs["nx_dir"] >> m_normal_x_dir;
            fs["ny_dir"] >> m_normal_y_dir;
            fs["nz_dir"] >> m_normal_z_dir;
            fs.release();
        }
        cv::Mat loadPeng8UC1( std::string file_name ) const
        {
            return cv::imread( m_peng_dir + file_name, CV_LOAD_IMAGE_GRAYSCALE );
        }
        cv::Mat loadRGB8UC3( std::string file_name ) const
        {
            return cv::imread( m_rgb_dir + file_name, CV_LOAD_IMAGE_COLOR );
        }
        cv::Mat loadLab8UC3( std::string file_name ) const
        {
            cv::Mat lab_8UC3, rgb_8UC3 = loadRGB8UC3(file_name);
            cv::cvtColor(rgb_8UC3, lab_8UC3, CV_BGR2Lab);
            return lab_8UC3;
        }
        cv::Mat loadLab32FC3( std::string file_name ) const
        {
            cv::Mat lab_32FC3 = loadLab8UC3(file_name);
            lab_32FC3.convertTo(lab_32FC3, CV_32FC3, 1.0/255);
            return lab_32FC3;
        }
        cv::Mat loadDepth32FC1( std::string file_name ) const
        {
            cv::Mat depth_16UC1 = cv::imread( m_depth_dir + file_name, CV_LOAD_IMAGE_UNCHANGED );
            cv::Mat depth_32FC1;
            depth_16UC1.convertTo( depth_32FC1, CV_32FC1, 0.001 );
            return depth_32FC1;
        }
        cv::Mat loadDepthRaw16UC1( std::string file_name ) const
        {
            return cv::imread( m_depth_raw_dir + file_name, CV_LOAD_IMAGE_UNCHANGED );
        }
        void disparity8UC1ToDepth32FC1( const cv::Mat &disp_8UC1, cv::Mat& depth) const
        {
            disp_8UC1.convertTo(depth, CV_32FC1);
            
            cv::normalize(depth, depth, 1, 0, cv::NORM_INF);
            depth = 1 - depth;
            return;
            
            float* hist = new float[256];
            memset(hist, 0, sizeof(float)*256);
            int numPixels = disp_8UC1.rows*disp_8UC1.cols;
            for (int i=0; i<numPixels; ++i)
                hist[disp_8UC1.data[i]] += 1;
            
            double disp_threshold = 0.05;
            double c = 1;
            double d = 5;
            float acchist = 0;
            double a, b;
            for (int i=0; i<256; ++i)
            {
                acchist += hist[i];
                if (double(acchist)/numPixels>disp_threshold)			//top 50%
                {
                    b = i;
                    break;
                }
            }
            acchist = 0;
            for (int i=255; i>=0; --i)
            {
                acchist += hist[i];
                if (double(acchist)/numPixels>disp_threshold)			//top 50%
                {
                    a = i;
                    break;
                }
            }
            
//            double dTh = b;
//            depth.setTo(dTh,depth<dTh);
            
            //cv::normalize(depth, depth, 1, 0, cv::NORM_INF);
            
            double x = (a*b*d - a*b*c)/ (a-b);
            double y = (a*c-b*d) / (a-b);
            
//            std::cout << a << " " << b << " " << x << " " << y << std::endl;
            
            //depth = 1.0+4.0 /(depth);
            depth = x / depth + y;
            depth.setTo(0.0,disp_8UC1==0);
            return;
            
            cv::Mat tmp = depth.clone();
            
            // disp1
            //            depth = 1.0+7.0 / depth;
            //            tmp.copyTo(depth,tmp==0);
            //            return;
            
            // disp2
            depth = 4.0 / depth;
            tmp.copyTo(depth,tmp==0);
            return;
            
            // disp3
            depth = 12.0 / depth;
            tmp.copyTo(depth,tmp==0);
            return;
            
            // disp
            depth = 8.0 / depth;
            tmp.copyTo(depth,tmp==0);
            return;
            
            int noSampleValue = 0, shadowValue = 0;
            double baseline = 75;
            int F = 570;
            // disparity = baseline * F / z;
            
            float mult = (float)(baseline /*mm*/ * F /*pixels*/);
            
            depth.create( disp_8UC1.size(), CV_32FC1);
            depth = cv::Scalar::all( 0 );
            for( int y = 0; y < disp_8UC1.rows; y++ )
            {
                for( int x = 0; x < disp_8UC1.cols; x++ )
                {
                    int curDisp = disp_8UC1.at<unsigned char>(y,x);
                    if( curDisp > 50 )
                        depth.at<float>(y,x) = mult / curDisp;
                    depth.at<float>(y,x) = 1.0 - curDisp/255.0;
                }
            }
            //disp.convertTo(disp,CV_8UC1);
        }
        cv::Mat loadDepthJu16UC1( std::string file_name, double scale=4.0 ) const
        {
            cv::Mat depth_32FC1 = loadDepthJu32FC1(file_name);
            depth_32FC1.convertTo(depth_32FC1, CV_16UC1, scale*1000);
            return depth_32FC1;
        }
        cv::Mat loadDepthJu32FC1( std::string file_name ) const
        {
            cv::Mat disp_8UC1 = cv::imread( m_depth_dir + file_name, CV_LOAD_IMAGE_UNCHANGED );
            cv::Mat depth_32FC1;
            disparity8UC1ToDepth32FC1(disp_8UC1, depth_32FC1);
            return depth_32FC1;
        }
        cv::Mat loadDepth8UC1( std::string file_name ) const
        {
            cv::Mat depth_32FC1 = loadDepth32FC1(file_name);
            cv::normalize( depth_32FC1, depth_32FC1, 1, 0, cv::NORM_INF );
            depth_32FC1 = 1.0 - depth_32FC1;
            depth_32FC1.convertTo(depth_32FC1, CV_8UC1, 255);
            return depth_32FC1;
        }
        cv::Mat loadSegmentation32SC1( std::string file_name ) const
        {
            cv::Mat out = cv::imread( m_segmentation_dir + splitStr( file_name, '.' )[ 0 ] + ".png", CV_LOAD_IMAGE_ANYDEPTH );
            out.convertTo(out, CV_32SC1);
            std::cout << m_segmentation_dir + file_name << std::endl;
            return out;
        }
        cv::Mat loadBackgroundPrior32FC1( std::string file_name ) const
        {
            cv::Mat out = cv::imread( m_background_prior_dir + splitStr( file_name, '.' )[ 0 ] + ".png", CV_LOAD_IMAGE_GRAYSCALE );
            out.convertTo(out, CV_32FC1, 1.0/255);
            return out;
        }
        cv::Mat loadDepthPrior32FC1( std::string file_name ) const
        {
            std::cout << m_depth_prior_dir + splitStr( file_name, '.' )[ 0 ] + ".png" << std::endl;
            cv::Mat out = cv::imread( m_depth_prior_dir + splitStr( file_name, '.' )[ 0 ] + ".png", CV_LOAD_IMAGE_GRAYSCALE );
            out.convertTo(out, CV_32FC1, 1.0/255);
            return out;
        }
        cv::Mat loadOrientationPrior32FC1( std::string file_name ) const
        {
            cv::Mat out = cv::imread( m_orientation_prior_dir + splitStr( file_name, '.' )[ 0 ] + ".png", CV_LOAD_IMAGE_GRAYSCALE );
            out.convertTo(out, CV_32FC1, 1.0/255);
            return out;
        }
        cv::Mat loadNormals32FC3( std::string file_name ) const
        {
            cv::Mat nx = cv::imread( m_normal_x_dir + splitStr( file_name, '.' )[ 0 ] + ".png", CV_LOAD_IMAGE_GRAYSCALE );
            cv::Mat ny = cv::imread( m_normal_y_dir + splitStr( file_name, '.' )[ 0 ] + ".png", CV_LOAD_IMAGE_GRAYSCALE );
            cv::Mat nz = cv::imread( m_normal_z_dir + splitStr( file_name, '.' )[ 0 ] + ".png", CV_LOAD_IMAGE_GRAYSCALE );
            std::vector<cv::Mat> normals = {nx, ny, nz};
            cv::Mat out (nx.size(), CV_8UC3);
            std::vector<cv::Mat> out_v = {out};
            int from_to[] = { 0,0, 1,1, 2,2 };
            cv::mixChannels(normals, out_v, from_to, 3);
            out.convertTo(out, CV_32FC3, 1.0/255);
            return out;
        }
        cv::Mat loadRegionContrast32FC1( std::string file_name ) const
        {
            cv::Mat out = cv::imread( m_region_contrast_dir + splitStr( file_name, '.' )[ 0 ] + ".png", CV_LOAD_IMAGE_GRAYSCALE );
            out.convertTo(out, CV_32FC1, 1.0/255);
            return out;
        }
        cv::Mat loadNormals32FC2( std::string file_name ) const
        {
            std::string yml_name = splitStr( file_name, '.' )[ 0 ] + ".yml";
            cv::FileStorage fs( m_normals_dir + yml_name, cv::FileStorage::READ );
            cv::Mat out;
            fs["data"] >> out;
            fs.release();
            return out;
        }
        cv::Mat loadGroundTruth8UC1( std::string file_name ) const
        {
            std::string png_name = splitStr( file_name, '.' )[ 0 ] + ".png";
            return cv::imread( m_ground_truth_dir + png_name, CV_LOAD_IMAGE_GRAYSCALE );
        }
        cv::Mat loadGroundTruthBoundaries8UC1( std::string file_name ) const
        {
            std::string png_name = splitStr( file_name, '.' )[ 0 ] + ".png";
            return cv::imread( m_ground_truth_boundaries_dir + png_name, CV_LOAD_IMAGE_GRAYSCALE );
        }
        bool isImage( std::string file_name ) const
        {
            int f_len = file_name.size();
            for ( int i = 0; i < m_image_ext.size(); i++ )
            {
                int i_len = m_image_ext.at( i ).size();
                if( f_len >= i_len )
                {
                    if ( file_name.substr( f_len - i_len, i_len).compare( m_image_ext.at( i ) ) == 0 )
                    {
                        return true;
                    }
                }
            }
            return false;
        }
        std::vector<std::string> getImagesInDirectory ( std::string dir_path = "" ) const
        {
            std::vector<std::string> out;
            DIR *dir;
            struct dirent *ent;
            if ( dir_path.size() == 0 )
            {
                dir_path = m_depth_dir;
            }
            if( ( dir = opendir( dir_path.c_str() ) ) != NULL ) {
                while( ( ent = readdir (dir) ) != NULL ) {
                    if ( isImage( ent->d_name ) ) {
                        out.push_back( ent->d_name );
                    }
                }
                closedir( dir );
            }
            return out;
        }
        bool isDir(std::string path)
        {
            struct stat buf;
            stat( path.c_str(), &buf );
            return S_ISDIR( buf.st_mode );
        }
        std::vector<std::string> getSubDirectories(std::string dir_path)
        {
            std::vector<std::string> out;
            DIR *dir;
            struct dirent *ent;
            if ((dir = opendir(dir_path.c_str())) != NULL)
            {
                while ((ent = readdir (dir)) != NULL)
                {
                    std::string name = ent->d_name;
                    if ( isDir( dir_path + name ) && name.compare( "." ) != 0 && name.compare( ".." ) != 0 )
                    {
                        out.push_back(name);
                    }
                }
                closedir (dir);
            }
            return out;
        }
        static void saveImage( std::string file_name, const cv::Mat &im, int type = -1 )
        {
            std::vector<int> compression_params;
            compression_params.push_back( CV_IMWRITE_PNG_COMPRESSION );
            compression_params.push_back( 0 );
            cv::Mat out;
            if ( type > 0 )
            {
                im.convertTo( out, type );
            }
            else
            {
                out = im;
            }
            
            cv::imwrite( file_name, out, compression_params );
        }
        static void saveImageResize( std::string file_name, const cv::Mat &im, int type = -1 )
        {
            std::vector<int> compression_params;
            compression_params.push_back( CV_IMWRITE_PNG_COMPRESSION );
            compression_params.push_back( 0 );
            cv::Mat out;
            if ( type > 0 )
            {
                im.convertTo( out, type );
            }
            else
            {
                out = im;
            }
            cv::resize(out,out,cv::Size(out.cols/2,out.rows/2));
            cv::imwrite( file_name, out, compression_params );
        }
        static void saveCSV(std::string file_name, std::vector<std::string> data) {
            std::ofstream file;
            file.open(file_name);
            for (int i=0; i<data.size(); i++) {
                file << data.at(i) << std::endl;
            }
            file.close();
        }
        
        void saveFillGraphSD(std::string file_name, std::vector<double> data)
        {
            saveDoubleVector(m_fill_graph_sd_dir + file_name, data);
        }
        void saveGapGraphSD(std::string file_name, std::vector<double> data)
        {
            saveDoubleVector(m_gap_graph_sd_dir + file_name, data);
        }
        void saveColourContrast(std::string file_name, std::vector<double> data)
        {
            saveDoubleVector(m_colour_contrast_dir + file_name, data);
        }
        std::vector<double> loadFillGraphSD(std::string file_name)
        {
            return loadDoubleVector(m_fill_graph_sd_dir + file_name);
        }
        std::vector<double> loadGapGraphSD(std::string file_name)
        {
            return loadDoubleVector(m_gap_graph_sd_dir + file_name);
        }
        std::vector<double> loadColourContrast(std::string file_name)
        {
            return loadDoubleVector(m_colour_contrast_dir + file_name);
        }
        
        static void saveDoubleVector(std::string file_path, std::vector<double> &data)
        {
            std::string yml_name = splitStr( file_path, '.' )[ 0 ] + ".yml";
            cv::FileStorage fs( yml_name, cv::FileStorage::WRITE );
            fs << "data" << cv::Mat(1, data.size(), CV_64FC1, data.data());
            fs.release();
        }
        std::vector<double> loadDoubleVector(std::string file_path)
        {
            cv::Mat tmp;
            std::string yml_name = splitStr( file_path, '.' )[ 0 ] + ".yml";
            cv::FileStorage fs( yml_name, cv::FileStorage::READ );
            fs["data"] >> tmp;
            fs.release();
//            std::vector<double> out ((double *)tmp.data, (double *)tmp.data+sizeof((double *)tmp.data)/sizeof(double));
            std::vector<double> out;
            for (int i=0; i<tmp.cols; i++)
            {
                out.push_back(tmp.at<double>(0,i));
            }
            return out;
        }
        static std::vector<std::string> splitStr( const std::string &s, char delim ) {
            std::stringstream ss( s );
            std::string item;
            std::vector<std::string> tokens;
            while ( std::getline( ss, item, delim ) )
            {
                tokens.push_back( item );
            }
            return tokens;
        }
        
        
        // input directories
        std::string m_ground_truth_dir;
        std::string m_ground_truth_boundaries_dir;
        std::string m_rgb_dir;
        std::string m_depth_dir;
        std::string m_depth_raw_dir;
        std::string m_segmentation_dir;
        std::string m_seg_dir;
        std::string m_normals_dir;
        std::string m_output_dir;
        std::string m_results_dir;
        std::string m_peng_dir;
        std::string m_background_prior_dir;
        std::string m_depth_prior_dir;
        std::string m_orientation_prior_dir;
        std::string m_region_contrast_dir;
        std::string m_normal_x_dir;
        std::string m_normal_y_dir;
        std::string m_normal_z_dir;
        std::string m_image_dir;
        
        std::string m_colour_contrast_dir;
        std::string m_fill_graph_sd_dir;
        std::string m_gap_graph_sd_dir;
        
        // valid image extensions
        std::vector<std::string> m_image_ext;
    };
};


#endif /* defined(__RGBDSaliency__Resources__) */
