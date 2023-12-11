#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>

using namespace std;
using namespace cv;
/*
*参数说明：
*src:待检测的原图像
*rho:以像素为单位的距离分辨率，即距离r离散时的单位长度
*theat:以角度为单位的距离分辨率，即角度Θ离散时的单位长度
*Threshold:累加器阈值，参数空间中离散化后每个方格被通过的
		   累计次数大于该阈值，则该方格代表的直线被视为在
		   原图像中存在
*lines:检测到的直线极坐标描述的系数数组，每条直线由两个参
	   数表示，分别为直线到原点的距离r和原点到直线的垂线与
	   x轴的夹角Θ
*/
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
	for (int i = 0; i < houghMat_rows; i++)
	{
		for (int j = 0; j < houghMat_cols; j++)
		{
			if (houghMat.ptr<float>(i)[j] > Threshold)
			{
				//line保存直线到原点的距离和直线到坐标原点的垂线和x轴的夹角
				Vec2f line(i * rho, j * theat * CV_PI / 180);
				lines.push_back(line);
			}
		}
	}

}

void drawLine(Mat& img, vector<Vec2f> lines, double rows, double cols, Scalar scalar, int n)
{
	Point pt1, pt2;
	for (int i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0];//直线到坐标原点的距离
		float theat = lines[i][1];//直线到坐标原点的垂线和x轴的夹角
		double a = cos(theat);
		double b = sin(theat);
		double x0 = a * rho, y0 = b * rho;//直线与过坐标原点的垂线的交点
		double length = max(rows, cols);//突出高宽的最大值

		//计算直线上的一点
		pt1.x = cvRound(x0 + length * (-b));
		pt1.y = cvRound(y0 + length * (a));
		//计算直线上的另一点
		pt2.x = cvRound(x0 - length * (-b));
		pt2.y = cvRound(y0 - length * (a));
		while (pt1.x == pt2.x && pt1.y == pt2.y)
		{
			//计算直线上的另一点
			pt2.x = cvRound(x0 + length * (-b));
			pt2.y = cvRound(y0 + length * (a));
		}
		//两点绘制直线
		line(img, pt1, pt2, scalar, n);
	}
}

int main()
{
	//Mat test = imread("../d.png");
	Mat img = imread("D:/_jpg_all/Test/photo.jpg");
	Mat test;
	resize(img, test, Size(1000, 750));
	vector<Vec2f> lines;
	myHoughLines(test, 1, 1, 100, lines);
	for (int i = 0; i < lines.size(); i++)
	{
		cout << "直线为：" << endl << lines[i] << endl;
	}
	Mat testResult = Mat::zeros(test.size(), CV_8U);//在全黑的图像中画出直线
	//test.copyTo(testResult);//在原图上画出直线

	drawLine(testResult, lines, test.rows, test.cols, Scalar(255), 1);
	imshow("原图：", test);
	imshow("变换后的直线：", testResult);
	waitKey(0);
	return 0;
}
