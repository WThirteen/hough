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
	for (int i = 0; i < houghMat_rows; i++){
		for (int j = 0; j < houghMat_cols; j++){
			if (houghMat.ptr<float>(i)[j] > Threshold){
				//line����ֱ�ߵ�ԭ��ľ����ֱ�ߵ�����ԭ��Ĵ��ߺ�x��ļн�
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
	// ��Сͼ�� ���ڼ���
	float row = img1.rows;
	float col = img1.cols;
	float r = img1.channels();
	Mat gray;
	if (r > 1) {
		cvtColor(img1, gray, COLOR_BGR2GRAY);
	}
	gray.convertTo(gray, CV_32FC1);
	//ʹ��canny ��ֵ��
	Mat canny_img;
	//canny�����ͼ��ΪCV_8UC1��ת��ΪCV_32FC1
	Canny(img1, canny_img, 60, 120, 3);
	canny_img.convertTo(canny_img, CV_32FC1);
	//3Ϊsobel�˲�ģ���С
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
					// ͶƱ
					//if (temp > 0) {
					//	hough_mat.at<float>(round(temp), theta)++;
					//}
					hough_mat.at<float>(round(temp) + row1, theta)++;
				 }
			}
		}
	}
	//�����洢ֱ�߲����Ķ�ά����
	vector<Vec2f> lines;
	//�ȼ���vector<vector<float>> lines;
	vector<Vec2f> lines2;
	myHoughLines(img1, 1, 1, 100, lines2);
	int judge = 150;
	//����hongh����
	for (int i = 0; i < 2 * row1 - 2; i++) {
		for (int j = 0; j < 180 - 2; j++) {
			if (hough_mat.at<float>(i, j) > judge) {
				//���������ֵ��rou
				Vec2f point(i, j * PI / 180);
				//�˲�����3*3
				hough_mat.at<float>(i, j + 1) = 0;
				hough_mat.at<float>(i, j + 2) = 0;
				hough_mat.at<float>(i + 1, j) = 0;
				hough_mat.at<float>(i + 1, j + 1) = 0;
				hough_mat.at<float>(i + 1, j + 2) = 0;
				hough_mat.at<float>(i + 2, j) = 0;
				hough_mat.at<float>(i + 2, j + 1) = 0;
				hough_mat.at<float>(i + 2, j + 2) = 0;
				lines.push_back(point);
				//��i��j�Ļ����ƴ��
			}
		}
	}
	cout << "line1(�˲������)" << endl;

	for (vector<Vec2f>::iterator it = lines.begin(); it != lines.end(); it++) {
		Vec2f p = *it;
		cout << "ֱ�߲�����" << p << endl;
	}
	cout << "line2(plan B �����˲�����)" << endl;

	for (vector<Vec2f>::iterator it = lines2.begin(); it != lines2.end(); it++) {
		Vec2f p = *it;
		cout << "ֱ�߲�����" << p << endl;
	}
	//ʹ��line1����ͼ��
	//for (int i = 0; i < size(lines); i++) {
	//	draw_line(img1, lines[i][0], lines[i][1], row1);
	//}
	 
	//ʹ��line2����ͼ��
	//ʹ�ú���draw_line2
	for (int i = 0; i < size(lines2); i++) {
		draw_line2(img1, lines2[i][0], lines2[i][1], row1);
	}
	imshow("img1", img1);
	imshow("canny_img", canny_img);
	waitKey(0);
	return 0;
}