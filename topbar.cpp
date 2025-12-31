#include "topbar.h"
#include "pipeline.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

TopBar::TopBar(QWidget *parent)
    : QWidget(parent),
    layout (new QHBoxLayout),
    file_combo (new QComboBox),
    preset_combo (new QComboBox)
{
    setLayout(layout);
    layout->addWidget(file_combo);
    //layout->addWidget(preset_combo);

    file_combo->addItem("File");
    file_combo->addItem("Load Image");
    file_combo->addItem("Save Image");

    connect(file_combo, &QComboBox::currentTextChanged, this, [this] (const QString& q_string){
        std::string string = q_string.toStdString();
        if(string == "Load Image"){
            loadImage();
        }

        if(string == "Save Image"){
            saveImage();
        }
    });

    preset_combo->addItem("Dummy1");
}


void TopBar::loadImage(QString file_name){
    if(file_name.isEmpty()){
        file_name = QFileDialog::getOpenFileName(
            this, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)");

        if (file_name.isEmpty())
            return;
    }

    cv::Mat img = cv::imread(file_name.toStdString());
    pipeline->setImage(img);
    file_combo->setCurrentIndex(0);
}

void TopBar::saveImage(){
    QString path = QFileDialog::getSaveFileName(nullptr,
                                                 "Save Image", "assets/saved",
                                                 "Image Files (*.png *.jpg *.bmp *.jpeg *.PNG);;All Files (*)");
    if (!path.isEmpty() && !pipeline->final_image.empty()){
        cv::imwrite(path.toStdString(), pipeline->final_image);
    }
    file_combo->setCurrentIndex(0);
}

TopBar::~TopBar(){}
