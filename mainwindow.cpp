#include "mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
    main_widget (new QWidget),
    layout (new QVBoxLayout),
    bottom_layout (new QHBoxLayout),
    topbar (new TopBar(this)),
    sidebar (new SideBar(this)),
    viewer (new Viewer(this)),
    pipeline (new Pipeline())
{
    this->setWindowState(Qt::WindowMaximized);
    setCentralWidget(main_widget);
    main_widget->setLayout(layout);
    build_funcs();

    layout->addWidget(topbar, 10);
    layout->addLayout(bottom_layout, 90);
    bottom_layout->addWidget(sidebar, 20);
    bottom_layout->addWidget(viewer, 80);

    topbar->pipeline = pipeline;
    pipeline->viewer = viewer;

    connect(sidebar->run_button, &QPushButton::clicked, [this]{
        pipeline->processImage(sidebar->function_states);
    });

    topbar->loadImage("assets/sample-images/taylor.jpg");
    topbar->loadImage("assets/sample-images/taylor.jpg");

}

MainWindow::~MainWindow() {}
