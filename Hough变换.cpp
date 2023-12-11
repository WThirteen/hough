#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>

using namespace std;
using namespace cv;
#define PI 3.1415926

void myHoughLines(Mat& src, double rho, double theat, int Threshold, vector<Vec2f>& lines)
{
	if (src.empty() || rho < 0.1 || theat>360 || theat < 0)
		return;

	int row = src.rows;
	int col = src.cols;
	Mat gray;
	if (src.channels() > 1)
	{//灰度化
		cvtColor(src, gray, COLOR_BGR2GRAY);
	}
	else
		src.copyTo(gray);

	int maxDistance = sqrt(src.cols * src.cols + src.rows * src.rows);
	int houghMat_cols = 360 / theat;//霍夫变换后距离夹角坐标下对应的Mat的宽
	int houghMat_rows = maxDistance / rho;//霍夫坐标距离夹角下对应的Mat的高
	Mat houghMat = Mat::zeros(houghMat_rows, houghMat_cols, CV_32FC1);

	//边缘检测
	Canny(gray, gray, 100, 200, 3);

	//二值化
	threshold(gray, gray, 160, 255, THRESH_BINARY);

	//遍历二值化后的图像
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (gray.ptr<uchar>(i)[j] != 0)
			{
				/*从0到360度遍历角度，得到一组关于距离夹角的离散点，即得到
				一组关于经过当前点(i，j)按单位角度theat旋转得到的直线*/
				for (int k = 0; k < 360 / theat; k += theat)
				{
					double r = i * sin(k * CV_PI / 180) + j * cos(k * CV_PI / 180);
					if (r >= 0)
					{//直线到原点的距离必须大于0
						//获得在霍夫变换距离夹角坐标系下对应的Mat的行的下标
						int r_subscript = r / rho;

						//经过该直线的点数加1
						houghMat.at<float>(r_subscript, k) = houghMat.at<float>(r_subscript, k) + 1;
					}

				}
			}
		}
	}
	//经过直线的点数大于阈值，则视为在原图中存在该直线
	for (int i = 0; i < houghMat_rows; i++){
		for (int j = 0; j < houghMat_cols; j++){
			if (houghMat.ptr<float>(i)[j] > Threshold){
				//line保存直线到原点的距离和直线到坐标原点的垂线和x轴的夹角
				Vec2f line(i * rho, j * theat * CV_PI / 180);
				lines.push_back(line);
			}
		}
	}
}

void draw_line(Mat img,float rou,float theta,float row1) {
	Point p1, p2;
	rou = rou - row1;
	p1.x = round(rou * cos(theta) + row1 * (-sin(theta)));
	p1.y = round(rou * sin(theta) + row1 * (cos(theta)));
	p2.x = round(rou * cos(theta) - row1 * (-sin(theta)));
	p2.y = round(rou * sin(theta) - row1 * (cos(theta)));
	Scalar color(0, 255, 0);
	line(img, p1, p2, color, 2);
}
void draw_line2(Mat img, float rou, float theta, float row1) {
	Point p1, p2;
	p1.x = round(rou * cos(theta) + row1 * (-sin(theta)));
	p1.y = round(rou * sin(theta) + row1 * (cos(theta)));
	p2.x = round(rou * cos(theta) - row1 * (-sin(theta)));
	p2.y = round(rou * sin(theta) - row1 * (cos(theta)));
	Scalar color(0, 255, 0);
	line(img, p1, p2, color, 2);
}

int main() {
	Mat img = imread("D:/_jpg_all/Test/photo.jpg");
	Mat img1;
	resize(img, img1, Size(1000, 750));
	// 缩小图像 便于计算
	float row = img1.rows;
	float col = img1.cols;
	float r = img1.channels();
	Mat gray;
	if (r > 1) {
		cvtColor(img1, gray, COLOR_BGR2GRAY);
	}
	gray.convertTo(gray, CV_32FC1);
	//使用canny 二值化
	Mat canny_img;
	//canny输出的图像为CV_8UC1需转化为CV_32FC1
	Canny(img1, canny_img, 60, 120, 3);
	canny_img.convertTo(canny_img, CV_32FC1);
	//3为sobel滤波模板大小
	//cout << canny_img << endl;
	float row1 = round(sqrt(row * row + col * col));
	float col1 = 180;
	Mat hough_mat = Mat::zeros(row1*2, col1, CV_32FC1);
	//Mat theta = Mat::zeros(1, 181, CV_32FC1);
	//theta.at<float>(1, 1)++;

	int temp;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (canny_img.at<float>(i, j) != 0) {
				for (int theta = 0; theta < 180; theta++) {
					//temp = i * cos(theta * PI / 180) + j * sin(theta * PI / 180);
					temp = i * sin(theta * PI / 180) + j * cos(theta * PI / 180);
					// 投票
					//if (temp > 0) {
					//	hough_mat.at<float>(round(temp), theta)++;
					//}
					hough_mat.at<float>(round(temp) + row1, theta)++;
				 }
			}
		}
	}
	//创建存储直线参数的二维向量
	vector<Vec2f> lines;
	//等价于vector<vector<float>> lines;
	vector<Vec2f> lines2;
	myHoughLines(img1, 1, 1, 100, lines2);
	int judge = 150;
	//遍历hongh矩阵
	for (int i = 0; i < 2 * row1 - 2; i++) {
		for (int j = 0; j < 180 - 2; j++) {
			if (hough_mat.at<float>(i, j) > judge) {
				//保存大于阈值的rou
				Vec2f point(i, j * PI / 180);
				//滤波处理3*3
				hough_mat.at<float>(i, j + 1) = 0;
				hough_mat.at<float>(i, j + 2) = 0;
				hough_mat.at<float>(i + 1, j) = 0;
				hough_mat.at<float>(i + 1, j + 1) = 0;
				hough_mat.at<float>(i + 1, j + 2) = 0;
				hough_mat.at<float>(i + 2, j) = 0;
				hough_mat.at<float>(i + 2, j + 1) = 0;
				hough_mat.at<float>(i + 2, j + 2) = 0;
				lines.push_back(point);
				//将i与j的弧度制存放
			}
		}
	}
	cout << "line1(滤波处理后)" << endl;

	for (vector<Vec2f>::iterator it = lines.begin(); it != lines.end(); it++) {
		Vec2f p = *it;
		cout << "直线参数：" << p << endl;
	}
	cout << "line2(plan B 且无滤波处理)" << endl;

	for (vector<Vec2f>::iterator it = lines2.begin(); it != lines2.end(); it++) {
		Vec2f p = *it;
		cout << "直线参数：" << p << endl;
	}
	//使用line1绘制图像
	//for (int i = 0; i < size(lines); i++) {
	//	draw_line(img1, lines[i][0], lines[i][1], row1);
	//}
	 
	//使用line2绘制图像
	//使用函数draw_line2
	for (int i = 0; i < size(lines2); i++) {
		draw_line2(img1, lines2[i][0], lines2[i][1], row1);
	}
	imshow("img1", img1);
	imshow("canny_img", canny_img);
	waitKey(0);
	return 0;
}