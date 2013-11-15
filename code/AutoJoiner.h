/*
 * Description: A class provides methods for joining images
 * Programmer: Rajesh
 * Date: 4/14/2013
 */

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <stack>

#define round(x) ((x) < 0? std::ceil((x) - 0.5) : std::floor((x) + 0.5))

class AutoJoiner
{
public:
	static void joint(std::vector<cv::Mat> &imgs, cv::Mat &joiner, std::vector<int> &order = std::vector<int>())
	{
		int num_imgs = imgs.size();
		std::map<std::pair<int, int>, cv::Mat> tmp_matches, matches, culled_matches;
		cv::Mat tmp_is_matched, is_matched, culled_is_matched;
		std::vector<cv::Mat> imgs_gray(num_imgs); // grayscale version
		std::vector<cv::Mat> img_crs; // homogeneous coordinates of the corners of images. Each corner in one column
		std::vector<cv::Mat> sts, best_sts; // similarity transformation matrices for each image
		int canvas_rows, canvas_cols, best_cr, best_cc;
		std::vector<cv::Mat> g_imgs; // global version of imgs
		std::vector<cv::Mat> masks; // the corresponding global masks
		cv::Mat tmp;
		double energy, min_energy;
		std::vector<int> best_order;
		std::vector<int> new_idx(num_imgs);
		int num_isolated = 0;
		std::vector<double> c;
		std::map<std::pair<int, int>, std::vector<double>> weights;

		for (int i = 0; i < num_imgs; ++i)
			cv::cvtColor(imgs[i], imgs_gray[i], CV_BGR2GRAY);
		// initialization
		interMatch(imgs_gray, tmp_matches, tmp_is_matched);
		//std::cout << tmp_is_matched << std::endl;
		imgs_gray.clear();
		for (int i = 0; i < num_imgs; ++i)
		{
			if ((cv::countNonZero(tmp_is_matched.col(i)) > 0))
				new_idx[i] = i - num_isolated;
			else
			{
				++num_isolated;
				new_idx[i] = -1;
			}
		}
		num_imgs -= num_isolated;
		is_matched = cv::Mat::zeros(num_imgs, num_imgs, CV_32S);
		for (std::map<std::pair<int, int>, cv::Mat>::iterator it = tmp_matches.begin(); it != tmp_matches.end(); ++it)
		{
			matches[std::pair<int, int>(new_idx[(it->first).first], new_idx[(it->first).second])] = it->second;
			is_matched.at<int>(new_idx[(it->first).first], new_idx[(it->first).second]) =
				tmp_is_matched.at<int>((it->first).first, (it->first).second);
		}
		for (int i = 0; i < new_idx.size(); ++i)
		{
			if (new_idx[i] != -1)
				imgs[new_idx[i]] = imgs[i];
		}
		imgs.resize(num_imgs);
		for (int i = 0; i < num_imgs; ++i)
			order.push_back(i);
		img_crs.resize(num_imgs);
		for (int i = 0; i < num_imgs; ++i)
		{
			img_crs[i] = cv::Mat::ones(3, 4, CV_64F);
			img_crs[i].at<double>(0, 0) = 0; img_crs[i].at<double>(1, 0) = imgs[i].rows - 1; // lower-left
			img_crs[i].at<double>(0, 1) = imgs[i].cols - 1; img_crs[i].at<double>(1, 1) = imgs[i].rows - 1; // lower-right
			img_crs[i].at<double>(0, 2) = imgs[i].cols - 1; img_crs[i].at<double>(1, 2) = 0; // upper-right
			img_crs[i].at<double>(0, 3) = 0; img_crs[i].at<double>(1, 3) = 0; // upper-left
		}
		aline(img_crs, matches, is_matched, sts, c, &canvas_rows, &canvas_cols);
		// precompute global images and global masks to save some computation
		g_imgs.resize(num_imgs);
		masks.resize(num_imgs);
		for (int i = 0; i < num_imgs; ++i)
		{
			g_imgs[i].create(canvas_rows, canvas_cols, CV_8UC3);
			masks[i].create(canvas_rows, canvas_cols, CV_64F);
			tmp = cv::Mat::ones(imgs[i].size(), CV_64F) * 255; // white image
			cv::warpAffine(imgs[i], g_imgs[i], sts[i].rowRange(0, 2), g_imgs[i].size(),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
			cv::warpAffine(tmp, masks[i], sts[i].rowRange(0, 2), masks[i].size(),
				cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
			g_imgs[i].convertTo(g_imgs[i], CV_64FC3); // cast to double for the sake of computation
		}
		//assignWeights(matches, is_matched, masks, order, sts, weights);
		//ba(matches, is_matched, weights, c, order, img_crs, sts, &canvas_rows, &canvas_cols);
		// reorder the images to minimize cost in terms of cross-border gradients
			shuffle(g_imgs, masks, order);
		//min_energy = energy = computeEnergy(g_imgs, masks, order);
		//best_order = order;
		//best_sts.resize(sts.size());
		//for (int i = 0; i < sts.size(); ++i)
		//	sts[i].copyTo(best_sts[i]);
		//best_cr = canvas_rows;
		//best_cc = canvas_cols;
		//test
		joiner = cv::Mat::zeros(canvas_rows, canvas_cols, CV_8UC3);
		drawImgOnCanvas(imgs, sts, is_matched, joiner, order);
		//cv::imshow("test1", canvas);
		//cv::waitKey(0);
		//// main loop
		//for (int k = 0; k < 5; ++k)
		//{
		//	// reorder the images to minimize cost in terms of cross-border gradients
		//	shuffle(g_imgs, masks, order);
		//	canvas = cv::Mat::zeros(canvas_rows, canvas_cols, CV_8UC3);
		//	drawImgOnCanvas(imgs, sts, is_matched, canvas, order);
		//	cv::imshow("test1", canvas);
		//	cv::imwrite("surf_lake.jpg", canvas);
		//	cv::waitKey(0);
		//	// exclude the keypoint matches that are occluded or too far away from visible borders
		//	cullKeypoints(matches, is_matched, g_imgs, masks, order, sts, culled_matches, culled_is_matched);
		//	// adjust the images with the culled keypoint matches
		//	aline(img_crs, culled_matches, culled_is_matched, sts, c, &canvas_rows, &canvas_cols);
		//	// compute global images and global masks again since they are changed
		//	for (int i = 0; i < num_imgs; ++i)
		//	{
		//		g_imgs[i].create(canvas_rows, canvas_cols, CV_8UC3);
		//		masks[i].create(canvas_rows, canvas_cols, CV_64F);
		//		tmp = cv::Mat::ones(imgs[i].size(), CV_64F) * 255; // white image
		//		cv::warpAffine(imgs[i], g_imgs[i], sts[i].rowRange(0, 2), g_imgs[i].size(),
		//			cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		//		cv::warpAffine(tmp, masks[i], sts[i].rowRange(0, 2), masks[i].size(),
		//			cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
		//		g_imgs[i].convertTo(g_imgs[i], CV_64FC3); // cast to double for the sake of computation
		//	}
		//	// compute the energy for this iteration
		//	energy = computeEnergy(g_imgs, masks, order);
		//	if (energy < min_energy)
		//	{
		//		min_energy = energy;
		//		best_order = order;
		//		for (int i = 0; i < sts.size(); ++i)
		//			sts[i].copyTo(best_sts[i]);
		//		best_cr = canvas_rows;
		//		best_cc = canvas_cols;
		//	}
		//	// this two variable should be empty before they are passed to cullKeypoints
		//	culled_matches.clear();
		//	culled_is_matched = cv::Mat();
		//	canvas = cv::Mat::zeros(canvas_rows, canvas_cols, CV_8UC3);
		//	drawImgOnCanvas(imgs, sts, is_matched, canvas, order);
		//	cv::imshow("test1", canvas);
		//	cv::waitKey(0);
		//}
		////test
		//canvas = cv::Mat::zeros(best_cr, best_cc, CV_8UC3);
		//drawImgOnCanvas(imgs, best_sts, is_matched, canvas, best_order);
		//cv::imshow("test", canvas);
		//cv::waitKey(0);

		//return; // test
	}

private:
	static void ba(const std::map<std::pair<int, int>, cv::Mat> &matches, const cv::Mat is_matched,
		const std::map<std::pair<int, int>, std::vector<double>> &weights, std::vector<double> &c,
		const std::vector<int> &order, const std::vector<cv::Mat> &img_crs, std::vector<cv::Mat> &sts,
		int *canvas_rows = NULL, int *canvas_cols = NULL)
	{
		std::vector<int> sub_order;
		sub_order.push_back(order[0]);
		cv::Mat crs_mtx(3, order.size() * 4, CV_64F);

		for (int i = 1; i < order.size(); ++i)
		{
			sub_order.push_back(order[i]);
			lm(matches, is_matched, weights, c, sub_order, sts);
		}

		for (int i = 0; i < order.size(); ++i)
			crs_mtx.colRange(4*i, 4*i+4) = sts[i] * img_crs[i];
		double minx, miny, maxx, maxy;
		cv::minMaxIdx(crs_mtx.row(0), &minx, &maxx); cv::minMaxIdx(crs_mtx.row(1), &miny, &maxy);
		minx = std::floor(minx); miny = std::floor(miny); maxx = std::ceil(maxx); maxy = std::ceil(maxy); 
		cv::Mat offset = (cv::Mat_<double>(3, 1) << ((minx < 0)? -minx : 0), ((miny < 0)? -miny : 0), 0);
		if (minx < 0 || miny < 0)
		{
			for (int i = 0; i < order.size(); ++i)
				sts[i].col(2) += offset;
		}
		if (canvas_rows != NULL && canvas_cols != NULL)
		{
			*canvas_rows = maxy + ((miny < 0)? -miny : 0) + 1;
			*canvas_cols = maxx + ((minx < 0)? -minx : 0) + 1;
		}
	}

	// adjust image order[order.size() - 1] with respect to the images below it
	// H * kps1 -> kps2; R-1 = RT; t-1 = -t;
	static void lm(const std::map<std::pair<int, int>, cv::Mat> &matches, const cv::Mat is_matched,
		const std::map<std::pair<int, int>, std::vector<double>> &weights, std::vector<double> &c,
		const std::vector<int> &order, std::vector<cv::Mat> &sts)
	{
		cv::Mat r, Jx, Jy;
		cv::Mat rij, Jxij, Jyij;
		cv::Mat kpsi, kpsj;
		cv::Mat row_ones;
		cv::Mat theta(4, 1, CV_64F);
		int top = order[order.size() - 1]; // the image to adjust
		cv::Mat A;
		cv::Mat g;
		double e = 0, pre_e = std::numeric_limits<double>::max();
		std::vector<double> wv;
		cv::Mat norm_r;
		double m, v = 2.0;
		cv::Mat hlm;

		// the parameters of the similarity transform of image top
		theta.at<double>(0, 0) = c[top];
		theta.at<double>(1, 0) = std::acos(sts[top].at<double>(0, 0) / c[top]);
		theta.at<double>(2, 0) = sts[top].at<double>(0, 2);
		theta.at<double>(3, 0) = sts[top].at<double>(1, 2);
		// get r and Jacobian
		for (int j = 0; j < order.size() - 1; ++j)
		{
			if (is_matched.at<int>(top, order[j]) > 0)
			{
				wv = weights.find(std::pair<int, int>(top, order[j]))->second;
				kpsi = (matches.find(std::pair<int, int>(top, order[j]))->second).t();
				kpsj = kpsi.rowRange(2, 4);
				kpsi = kpsi.rowRange(0, 2);
				row_ones = cv::Mat::ones(1, kpsi.cols, kpsi.type());
				kpsi.push_back(row_ones);
				cv::Mat tmp = invAffine(sts[order[j]]);
				rij = (kpsj - tmp * sts[top] * kpsi).t();
				cv::pow(rij, 2.0, norm_r);
				norm_r = norm_r.col(0) + norm_r.col(1);
				for (int k = 0; k < norm_r.rows; ++k)
					e += wv[k] * errorFunc(norm_r.at<double>(k, 0), 5);
				computeJ(kpsi, theta.at<double>(0, 0), theta.at<double>(1, 0), tmp, Jxij, Jyij);
				r.push_back(rij);
				Jx.push_back(Jxij);
				Jy.push_back(Jyij);
			}
		}
		A = (Jx + Jy).t() * (Jx + Jy);
		g = (Jx + Jy).t() * (r.col(0) + r.col(1));
		cv::minMaxIdx(A.diag(), NULL, &m);
		// start iterations
		while (std::abs(pre_e - e) >= 1)
		{
			pre_e = e;
			cv::solve(A + m*cv::Mat::eye(A.rows, A.cols, A.type()), -1 * g, hlm);
			std::cout << hlm << std::endl;//
			cv::Mat tmp_theta = theta + hlm;
			cv::Mat d_tmp = 0.5 * hlm.t() * (m * hlm - g);
			double d = d_tmp.at<double>(0, 0);
			double theta1 = tmp_theta.at<double>(0, 0);
			double theta2 = tmp_theta.at<double>(1, 0);
			double theta3 = tmp_theta.at<double>(2, 0);
			double theta4 = tmp_theta.at<double>(3, 0);
			cv::Mat new_Hi = (cv::Mat_<double>(3, 3) << theta1 * std::cos(theta2), -theta1 * std::sin(theta2), theta3,  theta1 * std::sin(theta2), theta1 * std::cos(theta2), theta4, 0, 0, 1);
			r = cv::Mat();
			Jx = cv::Mat();
			Jy = cv::Mat();
			e = 0;
			// get r and Jacobian
			for (int j = 0; j < order.size() - 1; ++j)
			{
				if (is_matched.at<int>(top, order[j]) > 0)
				{
					wv = weights.find(std::pair<int, int>(top, order[j]))->second;
					kpsi = (matches.find(std::pair<int, int>(top, order[j]))->second).t();
					kpsj = kpsi.rowRange(2, 4);
					kpsi = kpsi.rowRange(0, 2);
					row_ones = cv::Mat::ones(1, kpsi.cols, kpsi.type());
					kpsi.push_back(row_ones);
					cv::Mat tmp = invAffine(sts[order[j]]);
					rij = (kpsj - tmp * new_Hi * kpsi).t();
					cv::pow(rij, 2.0, norm_r);
					norm_r = norm_r.col(0) + norm_r.col(1);
					for (int k = 0; k < norm_r.rows; ++k)
						e += wv[k] * errorFunc(norm_r.at<double>(k, 0), 5);
					computeJ(kpsi, theta1, theta2, tmp, Jxij, Jyij);
					r.push_back(rij);
					Jx.push_back(Jxij);
					Jy.push_back(Jyij);
				}
			}
			double q = (pre_e - e) / d;
			if (q > 0)
			{
				theta = tmp_theta;
				A = (Jx + Jy).t() * (Jx + Jy);
				g = (Jx + Jy).t() * (r.col(0) + r.col(1));
				m = m * std::max(1.0/3.0, 1 - std::pow(2 * q - 1, 3));
				v = 2.0;
			}
			else
			{
				m *= v;
				v *= 2;
			}
		}
		sts[top] = (cv::Mat_<double>(3, 3) << theta.at<double>(0, 0) * std::cos(theta.at<double>(1, 0)), -theta.at<double>(0, 0) * std::sin(theta.at<double>(1, 0)), theta.at<double>(2, 0),  theta.at<double>(0, 0) * std::sin(theta.at<double>(1, 0)), theta.at<double>(0, 0) * std::cos(theta.at<double>(1, 0)), theta.at<double>(3, 0), 0, 0, 1);
		c[top] = theta.at<double>(0, 0);
	}

	static double errorFunc(double x, double limit = -1)
	{
		return ((limit == -1)? x : std::min(std::abs(x), limit));
	}

	static cv::Mat invAffine(const cv::Mat &H, bool homo = false)
	{
		cv::Mat r_val = cv::Mat::zeros(3, 3, CV_64F);
		r_val.colRange(0, 2).colRange(0, 2) = H.rowRange(0, 2).colRange(0, 2).t();
		r_val.col(2) = H.col(2) * -1;
		r_val.at<double>(2, 2) = 1;
		return (homo? r_val : r_val.rowRange(0, 2));
	}

	// kps = [x0 x1 ... xn; y0 y1 ... yn; 1 1 ... 1]
	static void computeJ(const cv::Mat &kpsi, double theta1, double theta2, const cv::Mat &Hj_inv, cv::Mat &Jx, cv::Mat &Jy)
	{
		double a = std::cos(theta2);
		double b = std::sin(theta2);
		cv::Mat t1 = (cv::Mat_<double>(3, 3) << a, -b, 0, b, a, 0, 0, 0, 1);
		cv::Mat t2 = (cv::Mat_<double>(3, 3) << -theta1 * b, -theta1 * a, 0, theta1 * a, -theta1 * b, 0, 0, 0, 1);
		cv::Mat t3 = (cv::Mat_<double>(3, 3) << 0, 0, 1, 0, 0, 0, 0, 0, 1);
		cv::Mat t4 = (cv::Mat_<double>(3, 3) << 0, 0, 0, 0, 0, 1, 0, 0, 1);

		t1 = -1 * Hj_inv * t1 * kpsi;
		t2 = -1 * Hj_inv * t2 * kpsi;
		t3 = -1 * Hj_inv * t3 * kpsi;
		t4 = -1 * Hj_inv * t4 * kpsi;
		Jx.create(kpsi.cols, 4, CV_64F);
		Jy.create(kpsi.cols, 4, CV_64F);
		Jx.col(0) = t1.row(0).t();
		Jx.col(1) = t2.row(0).t();
		Jx.col(2) = t3.row(0).t();
		Jx.col(3) = t4.row(0).t();
		Jy.col(0) = t1.row(1).t();
		Jy.col(1) = t2.row(1).t();
		Jy.col(2) = t3.row(1).t();
		Jy.col(3) = t4.row(1).t();
	}

	static void assignWeights(const std::map<std::pair<int, int>, cv::Mat> &matches, const cv::Mat &is_matched,
		const std::vector<cv::Mat> &masks, const std::vector<int> &order, const std::vector<cv::Mat> &sts,
		std::map<std::pair<int, int>, std::vector<double>> &weights)
	{
		cv::Mat dx_kernel = (cv::Mat_<double>(1, 3) << -1, 0, 1);
		cv::Mat dy_kernel = (cv::Mat_<double>(3, 1) << -1, 0, 1);
		cv::Mat dx, dy;
		cv::Mat border = cv::Mat::zeros(masks[0].rows, masks[0].cols, masks[0].type());
		cv::Mat tmp_border;
		cv::Mat dst;
		cv::Mat kps1, kps2;
		double sigma2 = 2500;
		double w = 0.1;

		for (int i = 0; i < order.size(); ++i)
		{
			// border of image order[i]
			cv::filter2D(masks[order[i]], dx, masks[order[i]].depth(), dx_kernel);
			cv::filter2D(masks[order[i]], dy, masks[order[i]].depth(), dy_kernel);
			tmp_border = cv::abs(dx) + cv::abs(dy);
			tmp_border.copyTo(border, masks[order[i]] > 0);
			//cv::imshow("test", border);//
			//cv::waitKey(0);//
		}
		// get the distance to the nearest zero element
		cv::distanceTransform(border <= 0, dst, CV_DIST_L2, 3);
		//cv::imshow("test", dst);//
		//cv::waitKey(0);//
		// begin weight calculation
		for (int i = 0; i < is_matched.rows; ++i)
		{
			for (int j = 0; j < is_matched.cols; ++j)
			{
				if (is_matched.at<int>(i, j) > 0 && weights[std::pair<int, int>(i, j)].size() == 0)
				{
					kps1 = (matches.find(std::pair<int, int>(i, j))->second).t();
					kps2 = kps1.rowRange(2, 4);
					kps1 = kps1.rowRange(0, 2);
					cv::Mat row_ones = cv::Mat::ones(1, kps1.cols, kps1.type());
					kps1.push_back(row_ones);
					kps2.push_back(row_ones);
					kps1 = sts[i] * kps1; // global coordinates
					kps2 = sts[j] * kps2;
					std::vector<double> &wij = weights[std::pair<int, int>(i, j)];
					wij.resize(kps1.cols);
					// for each match, calculate its weight
					for (int k = 0; k < kps1.cols; ++k)
					{
						wij[k] = std::max(std::exp(-std::min(
							std::pow((double)dst.at<float>(round(kps1.at<double>(1, k)), round(kps1.at<double>(0, k))), 2.0),
							std::pow((double)dst.at<float>(round(kps2.at<double>(1, k)), round(kps2.at<double>(0, k))), 2.0))
							/ sigma2), w);
					}
					// since match is symmetric, copy wij to wji
					std::vector<double> &wji = weights[std::pair<int, int>(j, i)];
					wji.resize(kps1.cols);
					wji = wij;
				}
			}
		}
	}

	/*
	 * Cull a subset of keypoints that are not occluded and near to the visible border.
	 * Precondition: culled_matches and culled_is_matched should be empty.
	 */
	static void cullKeypoints(const std::map<std::pair<int, int>, cv::Mat> &matches, const cv::Mat &is_matched,
		const std::vector<cv::Mat> &g_imgs, const std::vector<cv::Mat> &masks, const std::vector<int> &order,
		const std::vector<cv::Mat> &sts, std::map<std::pair<int, int>, cv::Mat> &culled_matches, cv::Mat &culled_is_matched)
	{
		static const double THRESHOLD_DIST = 30.0;
		bool is_occluded;
		int idx, idx_small;
		int num_matched_imgs = order.size();
		cv::Mat kpsi, kpsj, g_kpsi, g_kpsj, m_kps, row_ones, kps_culled1, kps_culled2, tmp1, tmp2;
		cv::Mat dx, dy, dx_kernel = (cv::Mat_<double>(1, 3) << -1, 0, 1), dy_kernel = (cv::Mat_<double>(3, 1) << -1, 0, 1);
		cv::Mat tmp_border, border, m_kernel = cv::Mat::ones(3, 3, CV_64F);
		std::vector<int> order_inv(num_matched_imgs);
		for (int i = 0; i < num_matched_imgs; ++i)
			order_inv[order[i]] = i;
		is_matched.copyTo(culled_is_matched);
		for (int i = 0; i < culled_is_matched.rows; ++i)
		{
			for (int j = 0; j < culled_is_matched.cols; ++j)
			{
				if (culled_is_matched.at<int>(i, j) > 0)
				{
					idx = std::max(order_inv[i], order_inv[j]);
					idx_small = std::min(order_inv[i], order_inv[j]);
					kpsi = (matches.find(std::pair<int, int>(i, j))->second).t();
					kpsj = kpsi.rowRange(2, 4);
					kpsi = kpsi.rowRange(0, 2);
					row_ones = cv::Mat::ones(1, kpsi.cols, kpsi.type());
					kpsi.push_back(row_ones);
					kpsj.push_back(row_ones);
					g_kpsi = sts[i] * kpsi; // now they are in global coordinates
					g_kpsj = sts[j] * kpsj;
					m_kps = (g_kpsi.rowRange(0, 2) + g_kpsj.rowRange(0, 2)) / 2.0;
					tmp1 = cv::Mat();
					tmp2 = cv::Mat();
					tmp1.push_back(kpsi.rowRange(0, 2));
					tmp1.push_back(kpsj.rowRange(0, 2));
					tmp1 = tmp1.t();
					tmp2.push_back(kpsj.rowRange(0, 2));
					tmp2.push_back(kpsi.rowRange(0, 2));
					tmp2 = tmp2.t();
					kps_culled1 = cv::Mat();
					kps_culled2 = cv::Mat();
					cv::filter2D(masks[order[idx_small]], dx, masks[order[idx_small]].depth(), dx_kernel);
					cv::filter2D(masks[order[idx_small]], dy, masks[order[idx_small]].depth(), dy_kernel);
					tmp_border = cv::abs(dx) + cv::abs(dy);
					border = cv::Mat::zeros(tmp_border.size(), tmp_border.type());
					tmp_border.copyTo(border, masks[order[idx_small]] > 0);
					//cv::imshow("2", border); //
					cv::filter2D(masks[order[idx]], dx, masks[order[idx]].depth(), dx_kernel);
					cv::filter2D(masks[order[idx]], dy, masks[order[idx]].depth(), dy_kernel);
					tmp_border = cv::abs(dx) + cv::abs(dy);
					tmp_border.copyTo(border, masks[order[idx]] > 0);
					//cv::imshow("3", border); //
					border.setTo(cv::Scalar(0.0), masks[order[idx]] <= 0);
					//cv::imshow("4", border); //
					border.setTo(cv::Scalar(0.0), masks[order[idx_small]] <= 0);
					//cv::imshow("5", border); //
					for (int k = idx + 1; k < num_matched_imgs; ++k)
					{
						border.setTo(cv::Scalar(0.0), masks[order[k]] > 0);
						//cv::imshow("mask", masks[order[k]]);//
						//cv::imshow("7", border); //
						//cv::waitKey(0);//
					}
					//cv::imshow("7", border); //
					cv::filter2D(masks[order[idx_small]] - masks[order[idx]] > 0, tmp_border, CV_64F, m_kernel);
					border.setTo(cv::Scalar(0.0), tmp_border <= 0);
					//cv::imshow("1", border); //
					if (cv::countNonZero(border) != 0) // the cross-border is visible
					{
						cv::distanceTransform(border <= 0, border, CV_DIST_L2, 3);
						//cv::imshow("dst", border);//
						//std::cout << "non-zero elements: " << cv::countNonZero(border) << std::endl;//
						//std::cout << "all elements: " << border.total() << std::endl;//
						//cv::waitKey(0); //
						// check if the keypoint matches between image i and j are
						// occluded by topper images
						for (int l = 0; l < m_kps.cols; ++l)
						{
							is_occluded = false;
							for (int k = idx + 1; k < num_matched_imgs; ++k)
							{
								if (masks[order[k]].at<double>(round(m_kps.at<double>(1, l)), round(m_kps.at<double>(0, l))) > 0)
								{
									is_occluded = true;
									break;
								}
							}
							if (!is_occluded &&
								border.at<float>(round(m_kps.at<double>(1, l)), round(m_kps.at<double>(0, l))) < THRESHOLD_DIST)
							{
								kps_culled1.push_back(tmp1.row(l));
								kps_culled2.push_back(tmp2.row(l));
							}
						}
						if (kps_culled1.rows != kps_culled2.rows)
						{
							std::cout << "ERROR!";
							system("pause");
						}
						if (kps_culled1.rows > 0) // not all keypoints are occluded
						{
							culled_matches[std::pair<int, int>(i, j)] = kps_culled1;
							culled_matches[std::pair<int, int>(j, i)] = kps_culled2;
							culled_is_matched.at<int>(i, j) = -kps_culled1.rows;
							culled_is_matched.at<int>(j, i) = -kps_culled2.rows;
						}
						else // all keypoints are occluded
						{
							culled_is_matched.at<int>(i, j) = 0;
							culled_is_matched.at<int>(j, i) = 0;
						}
					}
					else // cross-border is totally obscured
					{
						culled_is_matched.at<int>(i, j) = 0;
						culled_is_matched.at<int>(j, i) = 0;
					}
				}
			}
		}
		culled_is_matched *= -1;
	}

	/*
	 * Non-optimal re-ordering of images to minimize inconsistency.
	 * Precondition: the vector order must store the current order of images.
	 * The method returns the order of the images.
	 */
	static void shuffle(const std::vector<cv::Mat> &g_imgs, const std::vector<cv::Mat> &masks,
		std::vector<int> &order)
	{
		double energy, min_energy;
		int num_imgs = order.size();
		std::vector<int> new_order, best_order;
		new_order.push_back(order[0]);
		for (int i = 1; i < num_imgs; ++i)
		{
			new_order.push_back(order[i]);
			min_energy = std::numeric_limits<double>::max();
			for (int j = new_order.size() - 1; j >= 0; --j)
			{
				energy = computeEnergy(g_imgs, masks, new_order);
				if (energy < min_energy) // a better order
				{
					min_energy = energy;
					best_order = new_order;
				}
				if (j != 0)
					std::swap(new_order[j], new_order[j-1]);
			}
			new_order = best_order;
		}
		order = best_order;
	}

	/*
	 * Compute energy as the sum of gradients across visible border with the specified image order.
	 * @param
	 *    g_imgs - global version of imgs
	 *    masks - the corresponding global masks
	 *    order - if i < j, then image order[i] is placed blow image order[j]. order.size() <= num_imgs
	 * @return_val
	 *    energy of the current setting
	 */
	static double computeEnergy(const std::vector<cv::Mat> &g_imgs, const std::vector<cv::Mat> &masks,
		const std::vector<int> order)
	{
		cv::Mat canvas, border, dx, dy, tmp_border, tmp, gradients;
		cv::Mat dx_filter = (cv::Mat_<double>(1, 3) << -1, 0, 1); // filter for partial derivative along x direction
		cv::Mat dy_filter = (cv::Mat_<double>(3, 1) << -1, 0, 1); // filter for partial derivative along y direction
		// get golobal image and visible border
		g_imgs[order[0]].copyTo(canvas);
		cv::filter2D(masks[order[0]], dx, masks[order[0]].depth(), dx_filter);
		cv::filter2D(masks[order[0]], dy, masks[order[0]].depth(), dy_filter);
		border = cv::Mat::zeros(dx.size(), dx.type());
		tmp_border = cv::abs(dx) + cv::abs(dy);
		tmp_border.copyTo(border, masks[order[0]] > 0);
		for (int i = 1; i < order.size(); ++i)
		{
			g_imgs[order[i]].copyTo(canvas, masks[order[i]] > 0);
			cv::filter2D(masks[order[i]], dx, masks[order[i]].depth(), dx_filter);
			cv::filter2D(masks[order[i]], dy, masks[order[i]].depth(), dy_filter);
			tmp_border = cv::abs(dx) + cv::abs(dy);
			tmp_border.copyTo(border, masks[order[i]] > 0);
		}
		// compute energy as the sum of cross-border gradients
		cv::filter2D(canvas, dx, canvas.depth(), dx_filter);
		cv::filter2D(canvas, dy, canvas.depth(), dy_filter);
		gradients = cv::abs(dx) + cv::abs(dy);
		tmp = cv::Mat::zeros(gradients.size(), gradients.type());
		gradients.copyTo(tmp, border > 0);

		return cv::sum(cv::sum(tmp)).val[0];
	}

	/*
	 * Precondition: the canvas should be large enough and set to zero.
	 */
	static void drawImgOnCanvas(const std::vector<cv::Mat> &imgs, const std::vector<cv::Mat> &sts,
		const cv::Mat &is_matched, cv::Mat &canvas, const std::vector<int> &order = std::vector<int>())
	{
		if (order.size() == 0) // no order is specified. Draw with default order
		{
			for (int i = 0; i < imgs.size(); ++i)
			{
				cv::warpAffine(imgs[i], canvas, sts[i].rowRange(0, 2), canvas.size(),
					cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
			}
		}
		else // order is specified. Use the specified order
		{
			for (int i = 0; i < imgs.size(); ++i)
			{
				cv::warpAffine(imgs[order[i]], canvas, sts[order[i]].rowRange(0, 2), canvas.size(),
					cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
			}
		}
	}

	/*
	 * Calculate the similarity transformation matrix that transforms image i to image j. User needs to ensure that
	 * image i and j are a match.
	 */
	static void stitch(int i, int j, std::vector<cv::Mat> &sts, const std::map<std::pair<int, int>, cv::Mat> &matches,
		std::vector<double> &c)
	{
		cv::Mat kpsi = matches.find(std::pair<int, int>(i, j))->second;
		cv::Mat kpsj = kpsi.colRange(2, 4).t(); // keypoints of image j
		kpsi = kpsi.colRange(0, 2).t(); // keypoints of image i
		cv::Mat row_ones = cv::Mat::ones(1, kpsi.cols, kpsi.type());
		kpsi.push_back(row_ones); // homogeneous coordinates
		kpsj.push_back(row_ones);
		kpsi = sts[i] * kpsi; // combine previous transformation (if any)
		kpsj = sts[j] * kpsj;
		sts[i] = computeST(kpsi.rowRange(0, 2), kpsj.rowRange(0, 2), true, &c[i]) * sts[i];
	}

	static void stitch(int i, std::vector<int> &js, std::vector<cv::Mat> &sts, const std::map<std::pair<int, int>, cv::Mat> &matches,
		std::vector<double> &c)
	{
		cv::Mat kps1, kps2;

		for (int j = 0; j < js.size(); ++j)
		{
			cv::Mat kpsi = matches.find(std::pair<int, int>(i, js[j]))->second;
			cv::Mat kpsj = kpsi.colRange(2, 4).t(); // keypoints of image j
			kpsi = kpsi.colRange(0, 2).t(); // keypoints of image i
			cv::Mat row_ones = cv::Mat::ones(1, kpsi.cols, kpsi.type());
			kpsi.push_back(row_ones); // homogeneous coordinates
			kpsj.push_back(row_ones);
			kpsi = sts[i] * kpsi; // combine previous transformation (if any)
			kpsj = sts[js[j]] * kpsj;
			kpsi = kpsi.rowRange(0, 2).t();
			kpsj = kpsj.rowRange(0, 2).t();
			kps1.push_back(kpsi);
			kps2.push_back(kpsj);
		}
		sts[i] = computeST(kps1.t(), kps2.t(), true, &c[i]) * sts[i];
	}

	/*
	 * Calculate a list of similarity transformation matrices that transform images from their local coordinates
	 * to the alined global coordinates. Initial alinement means that no re-ordering is considered. This it done
	 * when sts is empty.
	 * @param
	 *    img_crs - a vector of image corners
	 *    matches - a map stores keypoint information for each match
	 *    is_matched - match relationship indicator
	 *    sts - returned list of similarity transformation matrices where sts[i]*img_crs[i] -> global coordinates of image i.
	 *       If sts is empty, it will be initialized using identity matrices. Otherwise, it's used directly.
	 */
	static void aline(const std::vector<cv::Mat> &img_crs, const std::map<std::pair<int, int>, cv::Mat> &matches,
		const cv::Mat &is_matched, std::vector<cv::Mat> &sts, std::vector<double> &c, int *canvas_rows = NULL, int*canvas_cols = NULL)
	{
		int num_imgs = img_crs.size();
		int num_isolated = 0, num_matches, max = 0;
		int t = 0;
		cv::Mat is_used(num_imgs, 1, CV_32S, cv::Scalar(0));
		cv::Mat crs_mtx(3, num_imgs * 4, CV_64F);
		bool is_separated = false;

		c.resize(num_imgs);
		if (sts.size() == 0) // if sts is empty, initialize it
		{
			sts.resize(num_imgs);
			for (int i = 0; i < num_imgs; ++i)
			{
				sts[i] = cv::Mat::eye(3, 3, CV_64F);
				num_matches = cv::countNonZero(is_matched.col(i));
				if (num_matches == 0)
					++num_isolated;
				else if (num_matches > max) // start from the one with most matches
				{
					max = num_matches;
					t = i;
				}
			}
		}
		else // if sts is not empty, use it directly
		{
			for (int i = 0; i < num_imgs; ++i)
			{
				num_matches = cv::countNonZero(is_matched.col(i));
				if (num_matches == 0)
					++num_isolated;
				else if (num_matches > max)
				{
					max = num_matches;
					t = i;
				}
			}
		}
		is_used.at<int>(t, 0) = 1; // the base image
		c[t] = 1.0;
		while (cv::sum(is_used).val[0] < num_imgs - num_isolated)
		{
			int idx = -1, max = 0;
			for (int i = 0; i < num_imgs; ++i)
			{
				if (is_used.at<int>(i, 0) == 0 && cv::countNonZero(is_matched.col(i)) > 0)
				{
					if (is_separated)
					{
						is_used.at<int>(i, 0) = 1;
						is_separated = false;
						continue;
					}
					for (int j = 0; j < num_imgs; ++j)
					{
						if (is_used.at<int>(j, 0) == 1 && is_matched.at<int>(i, j) > max)
						{
							max = is_matched.at<int>(i, j);
							idx = i;
							t = j;
						}
					}
				}
			}
			if (idx == -1)
			{
				is_separated = true;
				continue;
			}
			std::vector<int> js;
			for (int i = 0; i < num_imgs; ++i)
			{
				if (is_used.at<int>(i, 0) != 0 && is_matched.at<int>(idx, i) > 0)
					js.push_back(i);
			}
			stitch(idx, js, sts, matches, c); // stitch image idx to image t
			is_used.at<int>(idx, 0) = 1;
		}
		for (int i = 0; i < num_imgs; ++i)
			crs_mtx.colRange(4*i, 4*i+4) = sts[i] * img_crs[i];
		double minx, miny, maxx, maxy;
		cv::minMaxIdx(crs_mtx.row(0), &minx, &maxx); cv::minMaxIdx(crs_mtx.row(1), &miny, &maxy);
		minx = std::floor(minx); miny = std::floor(miny); maxx = std::ceil(maxx); maxy = std::ceil(maxy); 
		cv::Mat offset = (cv::Mat_<double>(3, 1) << ((minx < 0)? -minx : 0), ((miny < 0)? -miny : 0), 0);
		if (minx < 0 || miny < 0)
		{
			for (int i = 0; i < num_imgs; ++i)
				sts[i].col(2) += offset;
		}
		if (canvas_rows != NULL && canvas_cols != NULL)
		{
			*canvas_rows = maxy + ((miny < 0)? -miny : 0) + 1;
			*canvas_cols = maxx + ((minx < 0)? -minx : 0) + 1;
		}
	}

	/*
	 * Match each pair of images.
	 * @param
	 *    imgs - a vector of images to be matched
	 *    matches - a map of match infomation. matches.find(std::pair(i, j)) != matches.end()
	 *       if and only if imgs[i] is matched to imgs[j] and if imgs[i] is matched to imgs[j],
	 *       matches[std::pair(i, j)] will store the matrix that contains the coordinates of
	 *       the matched keypoints of the two images in the form [xi yi xj yj]
	 */
	static void interMatch(const std::vector<cv::Mat> &imgs, std::map<std::pair<int, int>, cv::Mat> &matches,
		cv::Mat &is_matched = cv::Mat())
	{
		int num_imgs = imgs.size();
		cv::Mat kps1, kps2;
		cv::Mat m_info1, m_info2; // coordinates of matched keypoints [xi yi xj yj]
		is_matched = cv::Mat(num_imgs, num_imgs, CV_32S, cv::Scalar(0));

		for (int i = 0; i < num_imgs - 1; ++i)
		{
			for (int j = i + 1; j < num_imgs; ++j)
			{
				kps1 = cv::Mat();
				kps2 = cv::Mat();
				match(imgs[i], imgs[j], kps1, kps2);
				if (ransac(kps1, kps2)) // image i is matched to image j
				{
					m_info1 = cv::Mat(); // force to detach from the last matrix
					m_info2 = cv::Mat();
					m_info1.create(kps1.rows, kps1.cols + kps2.cols, CV_64F);
					m_info2.create(kps1.rows, kps1.cols + kps2.cols, CV_64F);
					kps1.copyTo(m_info1.colRange(0, kps1.cols));
					kps2.copyTo(m_info1.colRange(kps1.cols, kps1.cols + kps2.cols));
					kps2.copyTo(m_info2.colRange(0, kps1.cols));
					kps1.copyTo(m_info2.colRange(kps1.cols, kps1.cols + kps2.cols));
					matches[std::pair<int, int>(i, j)] = m_info1;
					matches[std::pair<int, int>(j, i)] = m_info2;
					is_matched.at<int>(i, j) = kps1.rows; // the number of matches
					is_matched.at<int>(j, i) = kps1.rows;
				}
			}
		}
		return;
	}

	/*
	 * RAndom SAmple Consensus (RANSAC)
	 * Basic Procedures:
	 *    1. Randomly choose a minimal set of data points as a consensus set;
	 *    2. Compute the model parameters based on current consensus set;
	 *    3. Calculate the distance from each data point to the model. If the distance is smaller
	 *       than the threshold, the point is considered as an inlier and added to the consensus
	 *       set. Otherwise, the point is an outlier and thrown away;
	 *    4. If the size of current consensus set is greater than d, evaluate the error of current
	 *       model. If the error is smaller than the best_error, note down the error, model, and
	 *       consensus set as the best;
	 *    5. Repeat 1-4 k times.
	 * A Variant: usually, outliers are scarce relative to inliers. Thus the optimization criterion
	 *    can be changed to finding the maximum size of consensus set.
	 * @param
	 *    n - the minimum number of data required to fit the model
	 *    k - the number of iterations
	 *    t - a threshold for determination of in/outliers
	 *    d - the number of close data values required to assert that a model fits well to data
	 *    pts1 - a set of points to be transformed ([x0 y0; x1 y1; ... xn yn])
	 *    pts2 - the destination of transformation ([x0 y0; x1 y1; ... xn yn])
	 * @return_val
	 *    true - if a good similarity transform (model) is found
	 *    false - otherwise
	 * Note: the model is consider as the similarity transform from pts1 to pts2.
	 *       pts1 and pts2 should have the same size.
	 */
	static bool ransac(cv::Mat &pts1, cv::Mat &pts2, int n = 3, int k = 500, double t = 400, int d = 20)
	{
		if (pts1.rows <= d) // not enough data points to assert that a model is good
			return false;

		int d_copy = d;
		bool isGoodRand;
		cv::Mat cs(n, 1, CV_32S); // consensus set
		cv::Mat H; // the similarity transform matrix H = [cR, t]
		cv::Mat homo_pts1; // pts1 in homogeneous coordinates
		cv::Mat pts1_picked, pts2_picked;
		cv::Mat trans_pts1; // pts1 transformed to pts2's coordinate frame
		cv::Mat in_mark;

		homo_pts1.create(3, pts1.rows, pts1.type());
		homo_pts1.rowRange(0, 2) = pts1.t();
		homo_pts1.row(2) = cv::Scalar(1);

		while (k--)
		{
			isGoodRand = false;
			while (!isGoodRand)
			{
				isGoodRand = true;
				cv::randu(cs, cv::Scalar(0), cv::Scalar(pts1.rows)); // upper limit exclusive
				for (int i = 0; i < cs.rows - 1; ++i)
				{
					for (int j = i + 1; j < cs.rows; ++j)
					{
						if (cs.at<int>(i, 0) == cs.at<int>(j, 0))
							isGoodRand = false;
					}
				}
			}
			pts1_picked = cv::Mat();
			pts2_picked = cv::Mat();
			for (int i = 0; i < cs.rows; ++i)
			{
				pts1_picked.push_back(pts1.row(cs.at<int>(i, 0)));
				pts2_picked.push_back(pts2.row(cs.at<int>(i, 0)));
			}
			H = computeST(pts1_picked.t(), pts2_picked.t());
			trans_pts1 = H * homo_pts1; // [H*pt0 H*pt1 ... H*ptn]
			cv::pow(trans_pts1 - pts2.t(), 2.0, trans_pts1); // (H*pts1 - pts2).^2
			trans_pts1 = trans_pts1.row(0) + trans_pts1.row(1) < t; // inlier: 255, outlier: 0
			if (cv::countNonZero(trans_pts1) > d) // a better model
			{
				d = cv::countNonZero(trans_pts1);
				in_mark = trans_pts1;
			}
		}

		if (d > d_copy) // a model is found. Now d is the number of inliers
		{
			pts1_picked = cv::Mat();
			pts2_picked = cv::Mat();
			for (int i = 0; i < in_mark.cols; ++i)
			{
				if (in_mark.at<unsigned char>(0, i) != 0)
				{
					pts1_picked.push_back(pts1.row(i));
					pts2_picked.push_back(pts2.row(i));
				}
			}
			pts1 = pts1_picked;
			pts2 = pts2_picked;
			return true;
		}
		else // no suitable model
			return false;
	}

	/*
	 * Compute the similarity transformation matrix H = [cR, t].
	 * mv1 is the mean vector of pts1;
	 * mv2 is the mean vector of pts2;
	 * c11 is the variance of pts1;
	 * c22 is the variance of pts2;
	 * C12 is covariance matrix of pts1 and pts2;
	 * Let UDV' be the singular value decomposition of C12, then
	 * R = USV' where USV';
	 * t = mv2 - c*R*mv1;
	 * c = trace(DS)/c11 where S = I if det(U)det(V) = 1 or
	 * S = diag(1, ..., 1, -1) if det(U)det(V) = -1;
	 * Assume pts1 and pts2 have the format [x0 x1 ... xn; y0 y1 ... yn].
	 */
	static cv::Mat computeST(const cv::Mat &pts1, const cv::Mat &pts2, bool is_homo = false, double *scale = NULL)
	{
		cv::Mat mv1(2, 1, CV_64F), mv2(2, 1, CV_64F); // mean vectors
		cv::Scalar c11; // the variance of pts1
		cv::Mat C12; // covariance
		cv::Mat tmp1, tmp2; // temporary matrices used for calculation
		cv::Mat S;
		cv::Scalar c; // Scaling
		cv::Mat R, t; // Rotation and Translation
		cv::Mat H; // H = [cR, t]
		cv::Mat D; // a diagonal matrix used to store the singular values in svd.w

		mv1.row(0) = cv::mean(pts1.row(0));
		mv1.row(1) = cv::mean(pts1.row(1));
		mv2.row(0) = cv::mean(pts2.row(0));
		mv2.row(1) = cv::mean(pts2.row(1));
		cv::pow(pts1.row(0) - mv1.at<double>(0, 0), 2.0, tmp1); // tmp1 = (x - mx).^2
		cv::pow(pts1.row(1) - mv1.at<double>(1, 0), 2.0, tmp2); // tmp2 = (y - my).^2
		tmp1 += tmp2; // tmp1 = (x - mx).^2 + (y - my).^2
		c11 = cv::mean(tmp1); // c11 = sum((x - mx).^2 + (y - my).^2)/ n
		tmp1.create(pts1.size(), pts1.type());
		tmp2.create(pts2.size(), pts2.type());
		tmp1.row(0) = pts1.row(0) - mv1.at<double>(0, 0); // x - mx
		tmp1.row(1) = pts1.row(1) - mv1.at<double>(1, 0); // y - my; tmp = pts1 - mv1
		tmp2.row(0) = pts2.row(0) - mv2.at<double>(0, 0); // x - mx
		tmp2.row(1) = pts2.row(1) - mv2.at<double>(1, 0); // y - my; tmp = pts2 - mv2
		// View matrix multiplication as a sum of outer products!!
		tmp1 = tmp2 * tmp1.t(); // tmp1 = (pts2 - mv2)(pts1 - mv1)'
		C12 = tmp1 / pts1.cols;
		cv::SVD svd(C12); // compute the singular value decomposition of C12
		for (int i = 0; i < svd.w.rows - 1; ++i)
		{
			if (svd.w.at<double>(i, 0) < svd.w.at<double>(i+1, 0))
			{
				std::cout << "svd.w is not descendent!";
				exit(0);
			}
		}
		if (cv::determinant(svd.u)*cv::determinant(svd.vt) > 0) // det(U)*det(V) = 1
			S = cv::Mat::eye(2, 2, CV_64F);
		else // det(U)*det(V) = -1; det(V') = det(V)
			S = (cv::Mat_<double>(2, 2) << 1, 0, 0, -1);
		D = (cv::Mat_<double>(2, 2) << svd.w.at<double>(0, 0), 0, 0, svd.w.at<double>(1, 0));
		R = svd.u * S * svd.vt;
		c = cv::trace(D * S) / c11;
		t = mv2 - c.val[0] * R * mv1;
		H.create(R.rows, R.cols + t.cols, CV_64F);
		H.colRange(0, R.cols) = c.val[0] * R;
		t.copyTo(H.colRange(R.cols, H.cols));
		if (is_homo)
		{
			cv::Mat rv = (cv::Mat_<double>(1, 3) << 0, 0, 1);
			H.push_back(rv);
		}
		if (scale != NULL)
			*scale = c.val[0];

		return H;
	}

	/*
	 * Find the SIFT keypoints of two images and match them.
	 * @param
	 * img1 - the first image
	 * img2 - the second image
	 * pts1 - the matched keypoints from img1 represented by their local coordinates (row, col)
	 * pts2 - the matched keypoints from img2 represented by their local coordinates (row, col)
	 * @return_val
	 * - the number of matches
	 */
	static int match(const cv::Mat &img1, const cv::Mat &img2, cv::Mat &pts1, cv::Mat &pts2)
	{
		/* Important Note: always include the header files and call initModule_modulename() before
		 * using anything that derived from cv::Algorithm. Otherwise, the create method will return
		 * NULL since the linker thinks the module is not used and throws it away.
		 *
		 * cv::FeatureDetector is an abstract class that allows us to use different feature
		 * detection algorithms in a uniform format.
		 * To use this OpenCV feature, we need:
		 * 1. Use cv::FeatureDetector::create("AlgorithmName") to create a feature detector.
		 *    The method returns an OpenCV pointer cv::Ptr<FeatureDetector>;
		 * 2. To detect the features of an image, we use cv::FeatureDetector::detect(
		 *    const cv::Mat &image, std::vector<KeyPoint> &keypoints,
		 *    const cv::Mat &mask = cv::Mat()) where the mask is optional.
		 *
		 * cv::DescriptorExtractor is an abstract class that allows us to extract descriptors
		 * for keypoints using different algorithms in a uniform format.
		 * To use this OpenCV feature, we need:
		 * 1. Use cv::DescriptorExtractor::create("AlgorithmName") to create a descriptor
		 *    extractor. The method returns an OpenCV pointer cv::Ptr<DescriptorExtractor>;
		 * 2. To get the descriptors, use cv::DescriptorExtractor::compute(
		 *    const Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors) where the
		 *    descriptors is a matrix whose rows are descriptors of the provided keypoints.
		 *
		 * cv::DescriptorMatcher is an abstract class that allows us to match image keypoints
		 * using different algorithm in a uniform format.
		 * To use this OpenCV feature, we need:
		 * 1. Use cv::DescriptorMatcher::create(const string &type) to create a matcher where
		 *    type can be "BruteForce" (it uses L2 norm), "BruteForce-L1", "BruteForce-Hamming",
		 *    "BruteForce-Hamming(2)", or "FlannBased". The method returns an OpenCV pointer
		 *    cv::Ptr<DescriptorMatcher>;
		 * 2. Use cv::DescriptorMatcher::match(const cv::Mat &queryDescriptors,
		 *    const cv::Mat &trainDescriptors, std::vector<DMatch> &matches,
		 *    const cv::Mat &mask = cv::Mat()) to match the descriptors in queryDescriptors with
		 *    the descriptors in trainDescriptors.
		 */
		cv::SIFT sift;
		//cv::SURF surf;
		cv::Ptr<cv::DescriptorMatcher> dm = cv::DescriptorMatcher::create("BruteForce"); // NORM_L2
		std::vector<cv::KeyPoint> kps1, kps2;
		std::vector<cv::vector<cv::DMatch>> matches;
		cv::Mat dcp1, dcp2, tmp_pt = cv::Mat(1, 2, CV_64F);
		int count = 0; // the number of valid matches

		sift(img1, cv::noArray(), kps1, dcp1);
		sift(img2, cv::noArray(), kps2, dcp2);
		//surf(img1, cv::noArray(), kps1, dcp1);
		//surf(img2, cv::noArray(), kps2, dcp2);
		dm->knnMatch(dcp1, dcp2, matches, 2);
		//std::vector<cv::DMatch> m;
		//for (int i = 0; i < matches.size(); ++i)
		//{
		//	if (matches[i][0].distance < 0.6 * matches[i][1].distance)
		//	{
		//		m.push_back(matches[i][0]);
		//	}
		//}
		//cv::Mat out;
		//cv::drawMatches(img1, kps1, img2, kps2, matches, out);
		//cv::imshow("test", out);
		//cv::waitKey(0);
		/*
		 * When using SIFT, we need to avoid mismatches. During experiment, I found that we
		 * cannot simply use the distance between discriptors as a criterion of determining
		 * whether a match is a valid match. It's possible that a match has small distance
		 * but it's a mismatch. To solve this problem, I found that if a match is valid, it
		 * tends to differ from other matches. That is, if keypoint i in image 1 and keypoint
		 * j in image 2 form a valid match, then the distance between their disciptors is
		 * much smaller than the distance between the discriptor of keypoint i in image 1 and
		 * the discriptors of other keypoints in image 2, vice versa. Thus if we accept a
		 * match if and only if the distance of the match is much smaller than the distance
		 * of the second best match with respect to the same keypoint in image 1, then the
		 * result of the experiment turns out good. Most mismatches are excluded while valid
		 * matches are retained.
		 */
		for (int i = 0; i < matches.size(); ++i)
		{
			if (matches[i][0].distance < 0.6 * matches[i][1].distance)
			{
				++count;
				tmp_pt.at<double>(0, 0) = kps1[matches[i][0].queryIdx].pt.x;
				tmp_pt.at<double>(0, 1) = kps1[matches[i][0].queryIdx].pt.y;
				pts1.push_back(tmp_pt);
				tmp_pt.at<double>(0, 0) = kps2[matches[i][0].trainIdx].pt.x;
				tmp_pt.at<double>(0, 1) = kps2[matches[i][0].trainIdx].pt.y;
				pts2.push_back(tmp_pt);
			}
		}

		return count;
	}
};