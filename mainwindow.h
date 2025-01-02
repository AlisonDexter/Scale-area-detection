#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include<opencv2/opencv.hpp>
#include<QLabel>
#include <vector>
QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
//使用opencv的命名空间
using namespace cv;
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    QImage imageCenter(QImage qimage, QLabel*qLabel);

private slots:
    void on_btn_LoadImage_clicked();

    void on_btn_OupImage_clicked();

private:
    Ui::MainWindow *ui;
    Mat srcImage;
};
#endif // MAINWINDOW_H
