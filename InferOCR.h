#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "itkNiftiImageIO.h"
#include "itkImageRegionConstIterator.h"
#include "itkRGBPixel.h"
#include "itkBinaryImageToShapeLabelMapFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkJPEGImageIOFactory.h"
#include "itkRGBToLuminanceImageFilter.h"

	
const unsigned int Dimension = 2;
typedef unsigned char GrayPixelType;
typedef float NetType;
typedef itk::Image<GrayPixelType, Dimension> GrayScaleType;
typedef itk::Image<NetType, Dimension> NetInputType;
typedef itk::ImageFileWriter<GrayScaleType> WriterType;
typedef itk::ImageFileReader<GrayScaleType> ReaderType;
		

class InferOCR {
public:
	explicit InferOCR();
	~InferOCR();
	bool Iswhite_bg(cv::Mat binary, unsigned int area);
	bool Is_line(GrayScaleType::Pointer input_itk);
	GrayScaleType::Pointer delete_line(GrayScaleType::Pointer input_itk);

	std::string int_to_string(int input);
	void recog(GrayScaleType::Pointer input_itk, cv::Mat input_mat, int h, int w, std::string img_path);
	GrayScaleType::Pointer Mat2itk_uc(cv::Mat input);
	cv::Mat itk2Mat(GrayScaleType::Pointer input);
	void preprocess(std::string img_path);
};
