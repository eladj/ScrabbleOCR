//#include "opencv2/text.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "tesseract/baseapi.h"
#include "leptonica/allheaders.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


cv::Mat preProcess(cv::Mat img);
cv::Mat bgrToBinaryMap(cv::Mat img_bgr);
std::vector<cv::Point2f> findCorners(cv::Mat bw);
void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center);
cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b);
cv::Mat perspectiveTransformation(cv::Mat img, std::vector<cv::Point2f> quad_pts);
std::vector< std::vector<cv::Point2f> > buildTilesGrid(int grid_cols, int grid_rows, int img_width, int img_height);
std::vector< std::vector<bool> > isGridOcuupied(cv::Mat img, std::vector< std::vector<cv::Point2f> > grid);
std::vector< std::vector<char> > buildLettersGrid(cv::Mat img_bgr,
	std::vector< std::vector<cv::Point2f> > grid_mtx, 
	std::vector< std::vector<bool> > grid_occupied);
void bwClearBorder(cv::Mat &image);


int main(int argc, char** argv) {
	// Load image from command line arguments
	char* imageName = argv[1];
	int plotLevel = 0;
	//char* imageName = "C:\\Users\\elad\\Documents\\code\\DigitalScrabble\\board_letters (2).jpg";
	Mat img_bgr, bw;
	img_bgr = imread(imageName, 1);
	if (argc < 2 || !img_bgr.data)
	{
		printf(" No image data \n ");
		return -1;
	}
	if (argc > 2) {
		plotLevel = *argv[2] - '0';
	}
	if (argc > 3) {
		printf(" Too much arguments. Expected image filename and plotLevel. \n ");
		return -1;
	}

	// Main Program
	img_bgr = preProcess(img_bgr); // Resize
	bw = bgrToBinaryMap(img_bgr);
	std::vector<cv::Point2f> corners; // 4 corners of the board
	corners = findCorners(bw);
	cv::Mat quad; // Board image after prespective transformation to square
	quad = perspectiveTransformation(img_bgr, corners);
	std::vector< std::vector<cv::Point2f> > grid_mtx;
	std::vector< std::vector<bool> > grid_occupied;
	int grid_cols = 4, grid_rows = 4;  // Amount of squares on board
	grid_mtx = buildTilesGrid(grid_cols, grid_rows, quad.cols, quad.rows);
	grid_occupied = isGridOcuupied(quad, grid_mtx);
	std::vector< std::vector<char> > grid_chars;
	grid_chars = buildLettersGrid(quad, grid_mtx, grid_occupied);

	// Print Board
	for (int i = 0; i < grid_rows; i++) {
		for (int j = 0; j < grid_cols; j++) {
			cout << grid_chars[i][j];
		}
		cout << endl;
	}

	// Plot
	if (plotLevel > 0) {
		namedWindow("bw", WINDOW_NORMAL);
		imshow("bw", bw);

		// Draw contours
		Mat drawing = img_bgr.clone();
		std::vector< cv::Scalar > colors = { Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(255, 255, 255) };
		for (int i = 0; i < corners.size(); i++)
		{
			circle(drawing, corners[i], 35, colors[i], -1);
		}
		namedWindow("Corners", WINDOW_NORMAL);
		imshow("Corners", drawing);

		// Plots grid
		Mat quad_draw = quad.clone();
		for (int i = 0; i < grid_rows; i++) {
			for (int j = 0; j < grid_cols; j++) {
				if (grid_occupied[i][j]) {
					circle(quad_draw, grid_mtx[i][j], 35, Scalar(0, 0, 255), -1);
				}
				else {
					circle(quad_draw, grid_mtx[i][j], 35, Scalar(0, 255, 0), -1);
				}
				std::stringstream ss;
				ss << i << "," << j;
				putText(quad_draw, ss.str(), grid_mtx[i][j], FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 0, 255), 2, 2);
			}
		}
		namedWindow("Warped", WINDOW_NORMAL);
		imshow("Warped", quad_draw);
		waitKey(0);

		//cv::imwrite("./quad.png", quad_draw);
	}
	return 0;
}

// Handles all preProcess after getting the original image.
// Currently only resize the image to a fixed number of megapixels.
// The original aspect ratio of the image remains the same.
cv::Mat preProcess(cv::Mat img) {
	// Resize
	double im_size = 8e6;
	double factor = cv::sqrt(im_size / (img.rows*img.cols));
	int interpolation;
	if (factor < 1) { interpolation = INTER_AREA; } // if shrink
	else { interpolation = INTER_LINEAR; } // if enlarge
	cv::resize(img, img, Size(), factor, factor);
	return img;
}

// Finds Red regions in the BGR image.
// Returns BW binary Mat, with only the Red area.
cv::Mat bgrToBinaryMap(cv::Mat img_bgr) {
	cv::Mat img_hsv, bw, bw1, bw2;
	// Switch to BGR to HSV
	cv::cvtColor(img_bgr, img_hsv, CV_BGR2HSV);
	cv::Size blur_kernel = Size(7, 7);
	cv::GaussianBlur(img_hsv, img_hsv, blur_kernel, 0, 0);
	cv::Scalar low1 = cv::Scalar(0, 50, 50);
	cv::Scalar high1 = cv::Scalar(9, 230, 235);
	cv::Scalar low2 = cv::Scalar(170, 50, 50);
	cv::Scalar high2 = cv::Scalar(179, 230, 235);
	cv::inRange(img_hsv, low1, high1, bw1);
	cv::inRange(img_hsv, low2, high2, bw2);
	bw = bw1 | bw2;
	// Create a structuring element (SE)
	int morph_size = 5;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
		Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
	cv::morphologyEx(bw, bw, cv::MORPH_OPEN, element);
	cv::morphologyEx(bw, bw, cv::MORPH_CLOSE, element);
	return bw;
}

// Finds 4 corners of board in the BW image.
std::vector<cv::Point2f> findCorners(cv::Mat bw) {
	// Find Contours and 4 Corners
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	std::vector<cv::Point2f> approxCurve;
	findContours(bw, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
	double epsilon = 15; // maximum distance between the original curve and its approximation
	std::vector<cv::Point2f> corners;
	for (int i = 0; i < contours.size(); i++) {
		if (fabs(contourArea(contours[i])) > 15000) {
			approxPolyDP(contours[i], approxCurve, epsilon, true);
			if (approxCurve.size() == 4) {  // check if contour has 4 edges
				for (int i = 0; i < approxCurve.size(); i++)
				{
					corners.push_back(approxCurve[i]);
				}
			}
		}
	}
	/*// Hough Transform - Detect 4 Corners
	cv::Canny(bw, bw, 1, 1, 3);
	std::vector<cv::Vec4i> lines;
	cv::HoughLinesP(bw, lines, 1, CV_PI / 180, 70, 30, 10);
	std::vector<cv::Point2f> corners;
	for (int i = 0; i < lines.size(); i++)
	{
	for (int j = i + 1; j < lines.size(); j++)
	{
	cv::Point2f pt = computeIntersect(lines[i], lines[j]);
	if (pt.x >= 0 && pt.y >= 0)
	corners.push_back(pt);
	}
	}
	cv::approxPolyDP(cv::Mat(corners), approxCurve,
	cv::arcLength(cv::Mat(corners), true) * 0.02, true); */
	// Get mass center
	cv::Point2f center(0, 0);
	for (int i = 0; i < corners.size(); i++)
		center += corners[i];

	center *= (1. / corners.size());
	sortCorners(corners, center);
	return corners;
}

// Sorts the 4 corners of the quadrilateral.
// The first one will be the top-left and then clockwise.
// The input is the currnt vector of 4 corners (unsorted) and
// the center of mass of the quadrilateral.
void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center)
{
	std::vector<cv::Point2f> top, bot;

	for (int i = 0; i < corners.size(); i++)
	{
		if (corners[i].y < center.y)
			top.push_back(corners[i]);
		else
			bot.push_back(corners[i]);
	}

	cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
	cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
	cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
	cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];

	corners.clear();
	corners.push_back(tl);
	corners.push_back(tr);
	corners.push_back(br);
	corners.push_back(bl);
}

// Finds intersection point of 2 lines
cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
	int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
	int x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

	if (float d = ((float)(x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
	{
		cv::Point2f pt;
		pt.x = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / d;
		pt.y = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / d;
		return pt;
	}
	else
		return cv::Point2f(-1, -1);
}

// Applys prespective transformation on the found quadilateral.
// Gets an image and 4 corners and returns the quadilateral
// ROI as square.
cv::Mat perspectiveTransformation(cv::Mat img, std::vector<cv::Point2f> corners) {
	// Define the destination image
	int quad_dst_size = 800; // result quad x and y in pixels
	cv::Mat quad = cv::Mat::zeros(quad_dst_size, quad_dst_size, CV_8UC3);

	// Corners of the destination image
	std::vector<cv::Point2f> quad_pts;
	quad_pts.push_back(cv::Point2f(0, 0));
	quad_pts.push_back(cv::Point2f((float)quad.cols, 0));
	quad_pts.push_back(cv::Point2f((float)quad.cols, (float)quad.rows));
	quad_pts.push_back(cv::Point2f(0, (float)quad.rows));

	// Get transformation matrix
	cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);

	// Apply perspective transformation
	cv::warpPerspective(img, quad, transmtx, quad.size());
	//cv::imshow("quadrilateral", quad);
	return quad;
}

// Returns 2D matrix of the board grid, where each item holds the pixel
// (x,y) location of the center of that grid point.
// The board grid is (rows,cols).
// Input: 
//	grid_cols, grid_rows are the size of the board grid (in tiles).
//	img_width, img_height are the dimensions of the board image in pixels.
std::vector< std::vector<cv::Point2f> > buildTilesGrid(int grid_cols, int grid_rows, int img_width, int img_height) {
	// Build Tiles grid	
	int const border_shift = 47;
	int const grid_x = 4;
	int const grid_y = 4;
	float tile_width = (float)(img_width - 2 * border_shift) / (float)grid_x;
	float tile_height = (float)(img_height - 2 * border_shift) / (float)grid_y;
	float origin_x = (float)border_shift + ((float)tile_width / 2);
	float origin_y = (float)border_shift + ((float)tile_height / 2);
	vector< vector<cv::Point2f> > grid_mtx;
	grid_mtx.resize(grid_y, vector<cv::Point2f>(grid_x, cv::Point2f(0, 0)));
	std::vector<cv::Point2f> grid_pts;
	for (int i = 0; i < grid_y; i++) {
		for (int j = 0; j < grid_x; j++) {
			Point2f pt = Point2f(origin_x + tile_width*j, origin_y + tile_height*i);
			grid_mtx[i][j] = pt;
		}
	}
	return grid_mtx;
}

// Returns 2D matrix of the board grid, where each item holds True/False
// if this tile is empty or has a letter.
// INPUT:
//  img - BGR image of the square grid.
//  grid_mtx - 2D matrix of center pixel of each tile (from function buildTilesGrid)
std::vector< std::vector<bool> > isGridOcuupied(cv::Mat img, std::vector< std::vector<cv::Point2f> > grid_mtx) {
	double tile_width = fabs(grid_mtx[1][1].x - grid_mtx[0][0].x);
	double tile_height = fabs(grid_mtx[1][1].y - grid_mtx[0][0].y);
	int grid_rows = grid_mtx.size();
	int grid_cols = grid_mtx[0].size();
	std::vector< std::vector<bool> > grid_occupied(grid_rows, std::vector<bool>(grid_cols, false));
	Mat img_hsv;
	cvtColor(img, img_hsv, CV_BGR2HSV);
	vector<Mat> hsv_planes;
	split(img_hsv, hsv_planes);
	//std::vector<cv::Point> occupied_tiles;
	float const tile_std_th = 30;
	for (int i = 0; i < grid_rows; i++) {
		for (int j = 0; j < grid_cols; j++) {
			cv::Point2f pt = grid_mtx[i][j];
			Mat roi = hsv_planes[0](Rect(pt.x - tile_width / 2 + 20,
				pt.y - tile_height / 2 + 20,
				tile_width - 20,
				tile_height - 20));
			cv::Scalar tile_mean, tile_std;
			meanStdDev(roi, tile_mean, tile_std);
			if (tile_std.val[0] > tile_std_th) {
				grid_occupied[i][j] = true;
				//occupied_tiles.push_back(cv::Point(i, j));
			}
		}
	}
	return grid_occupied;
}

// Returns a 2D matrix with the letter in each tile. '0' marks empty tile.
// INPUT:
//  img_bgr - BGR image
//  grid_mtx - 2D matrix of center pixel of each tile (from function buildTilesGrid)
//  grid_occupied - 2D matrix of true/false if tile is occupied (from function isGridOcuupied)
std::vector< std::vector<char> > buildLettersGrid(cv::Mat img_bgr,
	std::vector< std::vector<cv::Point2f> > grid_mtx, std::vector< std::vector<bool> > grid_occupied) {
	int grid_rows = grid_mtx.size();
	int grid_cols = grid_mtx[0].size();
	std::vector< std::vector<char> > grid_letters(grid_rows, std::vector<char>(grid_cols, '0')); // The return grid.	
	double tile_width = fabs(grid_mtx[1][1].x - grid_mtx[0][0].x);
	double tile_height = fabs(grid_mtx[1][1].y - grid_mtx[0][0].y);
	// Switch to BGR to HSV
	cv::Mat img_hsv;	
	cv::cvtColor(img_bgr, img_hsv, CV_BGR2HSV);
	std::vector<cv::Mat> hsv_channels;
	cv::split(img_hsv, hsv_channels);
	cv::Mat value = hsv_channels[2]; // Value map
	int const letter_bw_th = 150; 
	int const max_BINARY_value = 255;
	int morph_size = 5;
	//const char* char_whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	//Ptr<cv::text::OCRTesseract> ocr = cv::text::OCRTesseract::create(NULL, NULL, NULL, 3, 10);
	//Ptr<cv::text::OCRTesseract> ocr = cv::text::OCRTesseract::create();
	//	char_whitelist, tesseract::OEM_DEFAULT, tesseract::PSM_SINGLE_CHAR);
	//std::string outChar;
	// Initialize tesseract-ocr with English, without specifying tessdata path	
	/*if (tess->Init(NULL, "eng")) {
		fprintf(stderr, "Could not initialize tesseract.\n");
		exit(1);
	}*/
	std::stringstream fn;
	for (int i = 0; i < grid_rows; i++) {
		for (int j = 0; j < grid_cols; j++) {
			if (grid_occupied[i][j]) {
				cv::Point2f pt = grid_mtx[i][j];
				Mat tile_roi = value(Rect(pt.x - tile_width / 2 + 30,
										  pt.y - tile_height / 2 + 25,
										  tile_width - 30,
										  tile_height - 30));
				cv::Mat tile_bw;
				cv::threshold(tile_roi, tile_bw, letter_bw_th, max_BINARY_value, cv::THRESH_BINARY);
				cv::bitwise_not(tile_bw, tile_bw);
				cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
				cv::morphologyEx(tile_bw, tile_bw, cv::MORPH_OPEN, element);
				cv::morphologyEx(tile_bw, tile_bw, cv::MORPH_CLOSE, element);
				bwClearBorder(tile_bw);
				/// Find contours
				vector<vector<Point> > contours;
				vector<Vec4i> hierarchy;
				cv:Mat tile_bw_edges;
				cv::Canny(tile_bw, tile_bw_edges, 0, 1);

				// save image
				//fn.str(std::string());
				//fn.clear();
				//fn << "tile_" << i << j << "_canny.png";
				//cv::imwrite(fn.str(), tile_bw);

				findContours(tile_bw_edges.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
				cv::Rect rect = boundingRect(contours[0]);
				//cv::RotatedRect rect;
				//rect = minAreaRect(contours[0]);	
				//Mat M, rotated, cropped;
				// get angle and size from the bounding box
				//float angle = rect.angle;
				//Size rect_size = rect.size;				
				//if (rect.angle < -45.) {
				//	angle += 90.0;
				//	cv::swap(rect_size.width, rect_size.height);
				//}
				// get the rotation matrix
				//M = getRotationMatrix2D(rect.center, angle, 1.0);
				// perform the affine transformation
				//warpAffine(tile_roi, rotated, M, tile_roi.size(), INTER_CUBIC);
				// crop the resulting image
				//getRectSubPix(rotated, rect_size, rect.center, cropped);
				rect -= cv::Point(4, 4);
				rect += cv::Size(8, 8);
				cv::Mat tile2 = tile_roi(rect);
				cv::Mat tile_clone = tile2.clone();
				cv::resize(tile_clone, tile_clone, Size(), 0.5, 0.5);
				// save image
				//fn.str(std::string());
				//fn.clear();
				//fn << "tile_" << i << j << ".png";				
				//cv::imwrite(fn.str(), tile_clone);				
				// Get OCR result
				//PIX *pix = pixCreate(tile_clone.size().width, tile_clone.size().height, 8);
				//pix->data = (l_uint32*)tile_clone.data;
				tesseract::TessBaseAPI tess;					
				tess.Init(NULL, "eng");				
				tess.SetPageSegMode(tesseract::PSM_SINGLE_CHAR); // set page segmentaion for single charcter				
				tess.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
				tess.SetImage((uchar*)tile_clone.data, tile_clone.size().width, tile_clone.size().height, tile_clone.channels(), tile_clone.step1());
				//tess.SetImage(pix);
				//tess.Recognize(0);
				const char* outText = tess.GetUTF8Text();
				int conf = tess.MeanTextConf();
				int offset;
				float slope;
				tess.GetTextDirection(&offset, &slope);
				//cv::text::OCRTesseract ocr = cv::text::OCRTesseract::create(NULL, NULL, NULL, 3, 10);
				//ocr->run(tile_clone, outChar);
				printf("%d,%d, OCR:%c, Confidence:%d, Offset:%d, Slope:%.3f\n", i,j,outText[0], conf, offset, slope);
				grid_letters[i][j] = outText[0];
				// Destroy used object and release memory
				tess.End();
				//delete[] outText;
			}
		}
	}	
	return grid_letters;
}

uchar white(255);

// Clears any blob that is on the border of BW image
void bwClearBorder(cv::Mat &image){
	// do top and bottom row
	for (int y = 0; y < image.rows; y += image.rows - 1){
		uchar* row = image.ptr<uchar>(y);
			for (int x = 0; x < image.cols; ++x){
				if (row[x] == white){
					cv::floodFill(image, cv::Point(x, y), cv::Scalar(0), (cv::Rect*)0, cv::Scalar(), cv::Scalar(200));
				}
			}
	}
	// fix left and right sides
	for (int y = 0; y < image.rows; ++y){
		uchar* row = image.ptr<uchar>(y);
			for (int x = 0; x < image.cols; x += image.cols - 1){
				if (row[x] == white){
					cv::floodFill(image, cv::Point(x, y), cv::Scalar(0), (cv::Rect*)0, cv::Scalar(), cv::Scalar(200));
				}
			}
	}
}
