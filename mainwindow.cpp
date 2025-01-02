#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include <vector>
#include<math.h>
using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}


// 构建积分图，用于快速计算图像块的平方和
Mat computeIntegralImage(const Mat& img) {
    Mat integralImg;
    integral(img, integralImg, CV_64F); // 计算积分图，CV_64F支持高精度
    return integralImg;
}

// 利用积分图快速计算图像块的平方和
double computeBlockSum(const Mat& integralImg, int x, int y, int blockSize) {
    int halfBlockSize = blockSize / 2;

    // 确保索引不越界
    int x1 = max(x - halfBlockSize, 0);
    int y1 = max(y - halfBlockSize, 0);
    int x2 = min(x + halfBlockSize, integralImg.cols - 2); // 注意积分图有一列扩展
    int y2 = min(y + halfBlockSize, integralImg.rows - 2);

    double A = integralImg.at<double>(y1, x1);
    double B = integralImg.at<double>(y1, x2 + 1);
    double C = integralImg.at<double>(y2 + 1, x1);
    double D = integralImg.at<double>(y2 + 1, x2 + 1);

    return max(0.0, D - B - C + A);
}

// 非局部均值滤波（基于积分图优化）
Mat nonLocalMeansFilterOptimized(const Mat& img, int blockSize, int searchSize, double h) {
    Mat result = Mat::zeros(img.size(), img.type());
    Mat integralImg = computeIntegralImage(img); // 构建积分图
    int halfSearchSize = searchSize / 2;

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            double weightSum = 0.0;
            double intensitySum = 0.0;

            for (int dy = -halfSearchSize; dy <= halfSearchSize; ++dy) {
                for (int dx = -halfSearchSize; dx <= halfSearchSize; ++dx) {
                    int neighborX = x + dx, neighborY = y + dy;

                    // 确保邻域不越界
                    if (neighborX >= 0 && neighborX < img.cols && neighborY >= 0 && neighborY < img.rows) {
                        // 使用积分图快速计算图像块的平方和
                        double blockSum1 = computeBlockSum(integralImg, x, y, blockSize);
                        double blockSum2 = computeBlockSum(integralImg, neighborX, neighborY, blockSize);

                        // 计算两个块的距离
                        double distance = abs(blockSum1 - blockSum2);
                        double weight = exp(-distance / (h * h));

                        intensitySum += weight * img.at<uchar>(neighborY, neighborX);
                        weightSum += weight;
                    }
                }
            }
            result.at<uchar>(y, x) = cvRound(intensitySum / weightSum);
        }
    }
    return result;
}

//输入图片
void MainWindow::on_btn_LoadImage_clicked()
{
    QString imageFilePath = QFileDialog::getOpenFileName(this,tr("打开图片"),"D:/AAAA/data","(所有图像(*.jpg *.png *.bmp))");
    if (imageFilePath.isEmpty())
    {
        return;
    }
    srcImage = imread(imageFilePath.toStdString());
    cvtColor(srcImage,srcImage,CV_BGR2RGB);
    QImage displayImg = QImage(srcImage.data,srcImage.cols,srcImage.rows,srcImage.cols * srcImage.channels(),QImage::Format_RGB888);
    QImage disimage = imageCenter(displayImg, ui->lbl_show1);
    ui->lbl_show1->setPixmap(QPixmap::fromImage(disimage));
}

//图片居中
QImage MainWindow::imageCenter(QImage qimage,QLabel *qLabel)
{
    QImage image;
    QSize imageSize = qimage.size();
    QSize labelSize = qLabel->size();
    double dWidthRation = 1.0*imageSize.width()/labelSize.width();
    double dHeightRatio = 1.0*imageSize.height()/labelSize.height();
    if(dWidthRation>dHeightRatio)
    {
        image = qimage.scaledToWidth(labelSize.width());
    }else{
        image = qimage.scaledToHeight(labelSize.height());
    }
    return image;

}



void MainWindow::on_btn_OupImage_clicked()
{
    Mat find_gray;
    // 图片转灰度图
    cvtColor(srcImage, find_gray, COLOR_BGR2GRAY);
    Mat gray;
    //判断通道
    int r = srcImage.channels();
    //图像的行列数
    int row = srcImage.rows;
    int col = srcImage.cols;
//    cout << "row=" << row << "col=" << col << endl;
    Mat img1;
    float kk = 1;
    //根据原始图像的尺寸动态调整图像的大小，
    //使得处理后的图像长和宽均处于1000到1500像素之间
    if (row > 5000 || col > 5000) {
        kk = 4.5;
    }
    else if (row > 3000 || col > 3000) {
        kk = 3;
    }
    else if (row > 2000 || col > 2000) {
        kk = 2;
    }
    else if (row > 1000 || col > 1000) {
        kk = 1.5;
    }
//    cout << "kk=" << kk << endl;
    //裁剪图片，降低分辨率
    cv::resize(srcImage, img1, Size(round(col / kk), round(row / kk)));
    Mat img2 = img1.clone();
    int row1 = img1.rows;
    int col1 = img1.cols;
    if (r > 1) {
        cvtColor(img1, gray, COLOR_BGR2GRAY);
    }
    else {
        gray = img1.clone();
    }
    //1. 减小分辨率， 双线性差值算法
    cv::resize(gray, gray, Size(round(col / kk), round(row / kk)));
    //2. 非局部均值滤波
    // 设置参数
    int blockSize = 5;     // 图像块大小
    int searchSize = 15;   // 搜索窗口大小
    double h = 15.0;       // 平滑参数

    // 计时
    double t = (double)getTickCount();

    // 执行非局部均值滤波（优化版）
    Mat denoisedImg = nonLocalMeansFilterOptimized(gray, blockSize, searchSize, h);
    Mat rect_gray = denoisedImg.clone();


    // 结束计时
    t = ((double)getTickCount() - t) / getTickFrequency();


    // Shi-Tomasi 角点检测（使用 cornerMinEigenVal）
    Mat edges;
    Mat copy = gray.clone();
    cornerMinEigenVal(copy, edges, 3);  // 计算每个像素的最小特征值，3是窗口大小
    // 先进行阈值处理，将灰度图转换为二值图（0或255）
    Mat binaryEdges;
    double minVal, maxVal;
    minMaxLoc(edges, &minVal, &maxVal);  // 获取角点结果的最小值和最大值
    threshold(edges, binaryEdges, 0.01 * maxVal, 255, THRESH_BINARY);

    // 确保二值图是8位无符号整数类型
    binaryEdges.convertTo(binaryEdges, CV_8U);

    // 查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(binaryEdges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 5. 遍历轮廓并筛选可能的标尺区域
    Rect rulerRect;
    for (size_t i = 0; i < contours.size(); i++) {
        Rect boundingBox = boundingRect(contours[i]);
        float aspectRatio = (float)boundingBox.height / boundingBox.width;

        // 根据长宽比和大小筛选标尺区域
        if (aspectRatio > 5.0 && boundingBox.width > 10 && boundingBox.height > 50) {
            rulerRect = boundingBox;
            break;
        }
    }

    if (rulerRect.area() > 0) {
        // 6. 裁剪标尺区域
        Mat rulerRegion = srcImage(rulerRect);

        // 7. 提取标尺的刻度线
        Mat rulerGray;
        cvtColor(rulerRegion, rulerGray, COLOR_BGR2GRAY);
        Mat rulerEdges;
        Canny(rulerGray, rulerEdges, 50, 150);

        std::vector<Vec4i> lines;
        HoughLinesP(rulerEdges, lines, 1, CV_PI / 180, 50, 30, 10);

        // 8. 将标尺区域转换为 QImage
        // 转换颜色格式从 BGR 到 RGB
        cv::cvtColor(rulerRegion, rulerRegion, cv::COLOR_BGR2RGB);

        // 将 Mat 转换为 QImage
        QImage displayImg = QImage(rulerRegion.data, rulerRegion.cols, rulerRegion.rows, rulerRegion.step, QImage::Format_RGB888);

        // 9. 显示图像到 lbl_show2
        ui->lbl_show2->setPixmap(QPixmap::fromImage(displayImg));
    }

}

