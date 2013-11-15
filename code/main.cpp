#include "AutoJoiner.h"
#include <fstream>

int main(int argc, char** argv)
{
	std::vector<cv::Mat> imgs;
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\la.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\lb.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\lc.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\ld.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\le.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\lf.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("a.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("b.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("c.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("d.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("e.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("f.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\wa.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\wb.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\wc.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\wd.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\we.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\wf.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\wg.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\wh.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\wi.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\ha.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\hb.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\hc.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\hd.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\he.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\cb1.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\cb2.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\cb3.JPG", CV_LOAD_IMAGE_COLOR));
	//imgs.push_back(cv::imread("D:\\sysu\\Computer Vision and Pattern Recognition\\images\\cb4.JPG", CV_LOAD_IMAGE_COLOR));
	//cv::Mat kps1, kps2;

	cv::initModule_features2d();
	cv::initModule_nonfree();

	std::fstream fs;
	if (argc < 3)
	{
		std::cerr << "Error! Note enough arguments." << std::endl;
		return 1;
	}
	fs.open(argv[2], std::ios_base::in);
	if (!fs.is_open())
	{
		std::cerr << "Error! Cannot open file: " << argv[2] << std::endl;
		return 1;
	}
	std::string fn;
	for (int i = 0; i < std::atoi(argv[1]); ++i)
	{
		std::getline(fs, fn);
		imgs.push_back(cv::imread(fn, CV_LOAD_IMAGE_COLOR));
		if (imgs[i].rows == 0 && imgs[i].cols == 0)
		{
			std::cerr << "Error! Cannot read image: " << fn << std::endl;
			return 1;
		}
	}

	int dr, dc;
	for (int i = 0; i < imgs.size(); ++i)
	{
		dr = imgs[i].rows;
		dc = imgs[i].cols;
		while (dr > 500 || dc > 500)
		{
			dr /= 2;
			dc /= 2;
		}
		cv::resize(imgs[i], imgs[i], cv::Size(dc, dr));
	}

	cv::Mat joiner;
	AutoJoiner::joint(imgs, joiner);
	
	if (argc == 4)
		cv::imwrite(argv[3], joiner);
	else
		cv::imwrite("joiner.jpg", joiner);

	return 0;
}