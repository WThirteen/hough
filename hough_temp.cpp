#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>

using namespace std;
using namespace cv;
/*
*����˵����
*src:������ԭͼ��
*rho:������Ϊ��λ�ľ���ֱ��ʣ�������r��ɢʱ�ĵ�λ����
*theat:�ԽǶ�Ϊ��λ�ľ���ֱ��ʣ����ǶȦ���ɢʱ�ĵ�λ����
*Threshold:�ۼ�����ֵ�������ռ�����ɢ����ÿ������ͨ����
		   �ۼƴ������ڸ���ֵ����÷�������ֱ�߱���Ϊ��
		   ԭͼ���д���
*lines:��⵽��ֱ�߼�����������ϵ�����飬ÿ��ֱ����������
	   ����ʾ���ֱ�Ϊֱ�ߵ�ԭ��ľ���r��ԭ�㵽ֱ�ߵĴ�����
	   x��ļнǦ�
*/
void myHoughLines(Mat& src, double rho, double theat, int Threshold, vector<Vec2f>& lines)
{
	if (src.empty() || rho < 0.1 || theat>360 || theat < 0)
		return;

	int row = src.rows;
	int col = src.cols;
	Mat gray;
	if (src.channels() > 1)
	{//�ҶȻ�
		cvtColor(src, gray, COLOR_BGR2GRAY);
	}
	else
		src.copyTo(gray);

	int maxDistance = sqrt(src.cols * src.cols + src.rows * src.rows);
	int houghMat_cols = 360 / theat;//����任�����н������¶�Ӧ��Mat�Ŀ�
	int houghMat_rows = maxDistance / rho;//�����������н��¶�Ӧ��Mat�ĸ�
	Mat houghMat = Mat::zeros(houghMat_rows, houghMat_cols, CV_32FC1);

	//��Ե���
	Canny(gray, gray, 100, 200, 3);

	//��ֵ��
	threshold(gray, gray, 160, 255, THRESH_BINARY);

	//������ֵ�����ͼ��
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			if (gray.ptr<uchar>(i)[j] != 0)
			{
				/*��0��360�ȱ����Ƕȣ��õ�һ����ھ���нǵ���ɢ�㣬���õ�
				һ����ھ�����ǰ��(i��j)����λ�Ƕ�theat��ת�õ���ֱ��*/
				for (int k = 0; k < 360 / theat; k += theat)
				{
					double r = i * sin(k * CV_PI / 180) + j * cos(k * CV_PI / 180);
					if (r >= 0)
					{//ֱ�ߵ�ԭ��ľ���������0
						//����ڻ���任����н�����ϵ�¶�Ӧ��Mat���е��±�
						int r_subscript = r / rho;

						//������ֱ�ߵĵ�����1
						houghMat.at<float>(r_subscript, k) = houghMat.at<float>(r_subscript, k) + 1;
					}

				}
			}
		}
	}

	//����ֱ�ߵĵ���������ֵ������Ϊ��ԭͼ�д��ڸ�ֱ��
	for (int i = 0; i < houghMat_rows; i++)
	{
		for (int j = 0; j < houghMat_cols; j++)
		{
			if (houghMat.ptr<float>(i)[j] > Threshold)
			{
				//line����ֱ�ߵ�ԭ��ľ����ֱ�ߵ�����ԭ��Ĵ��ߺ�x��ļн�
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
		float rho = lines[i][0];//ֱ�ߵ�����ԭ��ľ���
		float theat = lines[i][1];//ֱ�ߵ�����ԭ��Ĵ��ߺ�x��ļн�
		double a = cos(theat);
		double b = sin(theat);
		double x0 = a * rho, y0 = b * rho;//ֱ���������ԭ��Ĵ��ߵĽ���
		double length = max(rows, cols);//ͻ���߿�����ֵ

		//����ֱ���ϵ�һ��
		pt1.x = cvRound(x0 + length * (-b));
		pt1.y = cvRound(y0 + length * (a));
		//����ֱ���ϵ���һ��
		pt2.x = cvRound(x0 - length * (-b));
		pt2.y = cvRound(y0 - length * (a));
		while (pt1.x == pt2.x && pt1.y == pt2.y)
		{
			//����ֱ���ϵ���һ��
			pt2.x = cvRound(x0 + length * (-b));
			pt2.y = cvRound(y0 + length * (a));
		}
		//�������ֱ��
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
		cout << "ֱ��Ϊ��" << endl << lines[i] << endl;
	}
	Mat testResult = Mat::zeros(test.size(), CV_8U);//��ȫ�ڵ�ͼ���л���ֱ��
	//test.copyTo(testResult);//��ԭͼ�ϻ���ֱ��

	drawLine(testResult, lines, test.rows, test.cols, Scalar(255), 1);
	imshow("ԭͼ��", test);
	imshow("�任���ֱ�ߣ�", testResult);
	waitKey(0);
	return 0;
}
