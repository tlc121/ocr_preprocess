#include "InferOCR.h"
#include <stack>
#include <iostream>
#include <memory>
#include <vector>
#include <time.h>
#include <numeric>
#include <functional>
#include <math.h>

bool judge(const std::pair<int, cv::Mat> a, const std::pair<int, cv::Mat> b);


InferOCR::InferOCR() {
}

InferOCR::~InferOCR() {}
//###############################################################################################################################
cv::Mat InferOCR::itk2Mat(GrayScaleType::Pointer input) {
	GrayScaleType::RegionType region = input->GetLargestPossibleRegion();
	GrayScaleType::SizeType size = region.GetSize();
	unsigned int h = size[1];
	unsigned int w = size[0];

	cv::Mat ret(h, w, CV_8UC1, cv::Scalar::all(0));
	GrayScaleType::IndexType index;
	for (int i = 0; i < h; i++) {
		unsigned char* data_mat = ret.ptr<unsigned char>(i);
		for (int j = 0; j < w; j++) {
			index[0] = j;
			index[1] = i;
			unsigned char value = input->GetPixel(index);
			data_mat[j] = value;
		}
	}
	return ret;
}

bool judge(const std::pair<int, cv::Mat> a, const std::pair<int, cv::Mat> b) {
	return a.first < b.first;
}



//###############################################################################################################################
GrayScaleType::Pointer InferOCR::Mat2itk_uc(cv::Mat input) {
	int h = input.size().height;
	int w = input.size().width;
	GrayScaleType::Pointer ret = GrayScaleType::New();
	GrayScaleType::IndexType start;
	start[0] = 0;
	start[0] = 0;
	GrayScaleType::SizeType size_gray;
	size_gray[0] = w;
	size_gray[1] = h;
	GrayScaleType::RegionType region_gray;
	region_gray.SetSize(size_gray);
	region_gray.SetIndex(start);
	//std::cout << w << std::endl;
	ret->SetRegions(region_gray);
	ret->Allocate();
	GrayScaleType::IndexType index;
	for (int i = 0; i < h; i++) {
		unsigned char* data_mat = input.ptr<unsigned char>(i);
		for (int j = 0; j < w; j++) {
			unsigned char value = data_mat[j];
			index[0] = j;
			index[1] = i;
			ret->SetPixel(index, value);
		}
	}
	return ret;
}

//###############################################################################################################################
GrayScaleType::Pointer InferOCR::delete_line(GrayScaleType::Pointer input) {
	std::vector<int> distribution_h;
	GrayScaleType::RegionType region = input->GetLargestPossibleRegion();
	GrayScaleType::SizeType size = region.GetSize();
	unsigned int h = size[1];
	unsigned int w = size[0];

	GrayScaleType::IndexType index;
	for (int i = 0; i < h; i++) {
		int summ = 0;
		for (int j = 0; j < w; j++) {
			index[0] = j;
			index[1] = i;
			summ += input->GetPixel(index);
		}
		distribution_h.push_back(summ);
	}
			
	float mean = 0.0;
	float summ = 0;
	int count = 0;
	float std_var = 0.0;

	for (int i = 0; i < distribution_h.size(); i++) {
		if (distribution_h[i] > 0) {
			summ += distribution_h[i];
			count++;
		}
	}

	mean = summ * 1.0 / count;

	summ = 0;
	for (int i = 0; i < distribution_h.size(); i++) {
		if (distribution_h[i] > 0) {
			summ += (distribution_h[i] - mean)* (distribution_h[i] - mean);
		}
	}

	std_var = sqrt(summ*1.0/count);

	for (int i = 0; i < distribution_h.size(); i++) {
		//delete the inline
		if (distribution_h[i] > mean + 2 * std_var) {
			for (int j = 0; j < w; j++) {
				index[0] = j;
				index[1] = i;
				input->SetPixel(index, 0);
			}
		}
	}

	return input;
		
}

//####################################################################################################
std::string InferOCR::int_to_string(int input) {
	char buf[32] = { 0 };
	snprintf(buf, sizeof(buf), "%u", input);
	std::string str = buf;
	return str;
}

//###############################################################################################################################
//see if image's background is white
bool InferOCR::Iswhite_bg(cv::Mat binary, unsigned int area) {
	unsigned int count_1 = 0;
	double ratio_1 = 0.0;
	for (int i = 0; i < binary.size().height; i++) {
		uchar* data_binary = binary.ptr<uchar>(i);
		for (int j = 0; j < binary.size().width; j++) {
			if (data_binary[j] == 255) {
				count_1 += 1;
			}
		}
	}
	ratio_1 = double(count_1) / double(area);
	if (ratio_1 < 0.5)
		return true;
	return false;
}
//###############################################################################################################################
bool InferOCR::Is_line(GrayScaleType::Pointer input_itk) {
	GrayScaleType::SizeType nImageSize = input_itk->GetLargestPossibleRegion().GetSize();
	typedef  itk::BinaryImageToShapeLabelMapFilter<GrayScaleType>  BinaryImageToShapeLabelMapFilterType;
	BinaryImageToShapeLabelMapFilterType::Pointer binaryImageToShapeLabelMapFilter = BinaryImageToShapeLabelMapFilterType::New();
	binaryImageToShapeLabelMapFilter->FullyConnectedOn();
	binaryImageToShapeLabelMapFilter->SetInput(input_itk);
	try {
		binaryImageToShapeLabelMapFilter->Update();
	}
	catch (itk::ExceptionObject& err) {
		std::cerr << "ExceptionObject Caught" << std::endl;
		std::cerr << err << std::endl;
		return  NULL;
	}


	//remove some region we don't need
	long nRegionNum = 0;
	NetInputType::Pointer input_ = NetInputType::New();
	for (unsigned int i = 0; i < binaryImageToShapeLabelMapFilter->GetOutput()->GetNumberOfLabelObjects(); i++) {
		BinaryImageToShapeLabelMapFilterType::OutputImageType::LabelObjectType* labelObject =
			binaryImageToShapeLabelMapFilter->GetOutput()->GetNthLabelObject(i);
		BinaryImageToShapeLabelMapFilterType::OutputImageType::RegionType inputRegion =
			labelObject->GetBoundingBox();
		BinaryImageToShapeLabelMapFilterType::OutputImageType::RegionType::IndexType start = inputRegion.GetIndex();
		BinaryImageToShapeLabelMapFilterType::OutputImageType::RegionType::SizeType size = inputRegion.GetSize();
		//get the boudingbox index of the label
		int minx = start[0];
		int miny = start[1];
		int width = size[0];
		int height = size[1];
		int maxx = start[0] + size[0];
		int maxy = start[1] + size[1];
		if (width > 0.5 * nImageSize[0])
			return true;
	}
		
	return false;
}


//###############################################################################################################################
void InferOCR::recog(GrayScaleType::Pointer input_itk, cv::Mat input_mat, int input_h, int input_w, std::string img_path) {
	//region props
	GrayScaleType::SizeType nImageSize = input_itk->GetLargestPossibleRegion().GetSize();
	typedef  itk::BinaryImageToShapeLabelMapFilter<GrayScaleType>  BinaryImageToShapeLabelMapFilterType;
	BinaryImageToShapeLabelMapFilterType::Pointer binaryImageToShapeLabelMapFilter = BinaryImageToShapeLabelMapFilterType::New();
	binaryImageToShapeLabelMapFilter->FullyConnectedOn();
	binaryImageToShapeLabelMapFilter->SetInput(input_itk);
	try {
		binaryImageToShapeLabelMapFilter->Update();
	}
	catch (itk::ExceptionObject& err) {
		std::cerr << "ExceptionObject Caught" << std::endl;
		std::cerr << err << std::endl;
		return  NULL;
	}

	//create a new mat to store the segmentaion image
	int height = input_mat.size().height;
	int width = input_mat.size().width;
	cv::Mat ret(height, int(width*1.1), CV_8UC1, cv::Scalar::all(0));
	std::vector<std::pair<int, cv::Mat>> roi_list;
	std::vector<int> width_list;
	//remove some region we don't need
	long nRegionNum = 0;
	for (unsigned int i = 0; i < binaryImageToShapeLabelMapFilter->GetOutput()->GetNumberOfLabelObjects(); i++) {
		BinaryImageToShapeLabelMapFilterType::OutputImageType::LabelObjectType* labelObject =
			binaryImageToShapeLabelMapFilter->GetOutput()->GetNthLabelObject(i);
		BinaryImageToShapeLabelMapFilterType::OutputImageType::RegionType inputRegion =
			labelObject->GetBoundingBox();
		BinaryImageToShapeLabelMapFilterType::OutputImageType::RegionType::IndexType start = inputRegion.GetIndex();
		BinaryImageToShapeLabelMapFilterType::OutputImageType::RegionType::SizeType size = inputRegion.GetSize();
		//get the boudingbox index of the label
		int minx = start[0];
		int miny = start[1];
		int width = size[0];
		int height = size[1];
		int maxx = start[0] + size[0];
		int maxy = start[1] + size[1];
		if (minx < 0.01 * nImageSize[0] || miny < 0.01 * nImageSize[1] || (maxx - minx > 0.3 * nImageSize[0]) || (maxx - minx < 2))
			continue;
		else {
			cv::Mat roi_area = input_mat(cv::Rect2i(minx, 0 , width, nImageSize[1]));
			roi_list.push_back(std::pair<int, cv::Mat>(minx, roi_area));
			width_list.push_back(width);
			cv::Mat rgb;
		}
	}

	float sum = std::accumulate(std::begin(width_list), std::end(width_list), 0.0);
	float mean_width = sum / width_list.size(); //mean_value of the width
			
	int index = 0;

	std::sort(roi_list.begin(), roi_list.end(), judge);
	for (int i = 0; i < roi_list.size(); i++) {
		// make linux prediction
		cv::Mat network_input = roi_list[i].second;
		int xmin = roi_list[i].first + index;
		int width = network_input.size().width;
		int height = network_input.size().height;
		if (width_list[i] >= 1.5 * mean_width) {
			cv::Mat region1 = network_input(cv::Rect2i(0, 0, width / 2, height));
			cv::Mat region2 = network_input(cv::Rect2i(width/2, 0, width-(width / 2), height));
			int xmax = xmin + width;
			int cut_point = xmin + width / 2;
			int w_temp = cut_point - xmin;
			for (int h = 0; h < height; h++) {
				unsigned char* data_mat = region1.ptr<unsigned char>(h);
				unsigned char* data_mat_ret = ret.ptr<unsigned char>(h);
				for (int w = xmin; w < xmin + w_temp; w++) {
					data_mat_ret[w] = data_mat[w-xmin];
				}
			}

			xmin += w_temp + 1;
			for (int h = 0; h < height; h++) {
				unsigned char* data_mat = region2.ptr<unsigned char>(h);
				unsigned char* data_mat_ret = ret.ptr<unsigned char>(h);
				for (int w = xmin; w < xmax+1; w++) {
					data_mat_ret[w] = data_mat[w-xmin];
				}
			}

			index += 2;
		}
		else {
			std::cout << xmin << std::endl;
			for (int h = 0; h < height; h++) {
				unsigned char* data_mat = network_input.ptr<unsigned char>(h);
				unsigned char* data_mat_ret = ret.ptr<unsigned char>(h);
				for (int w = xmin; w < xmin + width; w++) {
					data_mat_ret[w] = data_mat[w - xmin];
				}
			}
			index += 1;
		}
	}

	std::string img_name = img_path.substr(img_path.find_last_of('/') + 1, -1);
	std::cout << img_name << std::endl;
	cv::imwrite("/hdd/sd5/tlc/OCR/cpp_image/"+img_name, ret);
	return;
}

//###############################################################################################################################
void InferOCR::preprocess(std::string img_path) {
	int input_w = shape[0];
	int input_h = shape[1];
			
	cv::Mat img;
	img = cv::imread(img_path, 0);
	if (img.empty()) {
		LOG4CPLUS_WARN(COMM::MyLogger::getInstance()->m_rootLog, "Could not open image: " << img_path);
		LOG4CPLUS_WARN(COMM::MyLogger::getInstance()->m_rootLog, "Is it broken? ");
		output_OCR->image_broken = true;
		return output_OCR;
	}
	int h = img.size().height;
	int w = img.size().width;
	unsigned int area = h * w;


	cv::Scalar mean;
	cv::Scalar dev;

	cv::meanStdDev(img, mean, dev);
	float m = mean.val[0];
	float s = dev.val[0];

	float threshold = m;
	cv::Mat binary(h, w, CV_8UC1);
	for (int i = 0; i < binary.size().height; i++) {
		unsigned char* data_binary = binary.ptr<unsigned char>(i);
		unsigned char* data_gray = img.ptr<unsigned char>(i);
		for (int j = 0; j < binary.size().width; j++) {
			if (data_gray[j] <= threshold) {
				data_binary[j] = 255;
			}
			else {
				data_binary[j] = 0;
			}
		}
	}


	//see if its background is white, if yes, go on. if not change the strategy of thresholding
	if (!this->Iswhite_bg(binary, area))
	{
		threshold = m + 0.5 * s;
		std::cout << "its not white background" << std::endl;
		for (int i = 0; i < binary.size().height; i++) {
			unsigned char* data_binary = binary.ptr<unsigned char>(i);
			unsigned char* data_gray = img.ptr<unsigned char>(i);
			for (int j = 0; j < binary.size().width; j++) {
				if (data_gray[j] <= threshold) {
					data_binary[j] = 0;
				}
				else {
					data_binary[j] = 255;
				}
			}
		}
	}
	GrayScaleType::Pointer binary_itk = Mat2itk_uc(binary);
	clock_t start = clock();
	if (this->Is_line(binary_itk))
		binary_itk = this->delete_line(binary_itk);


	binary = this->itk2Mat(binary_itk);
	this->recog(binary_itk, binary, input_h, input_w, img_path);
	clock_t duration = (clock() - start) / 1000;
	return;
}

