cmake_minimum_required(VERSION 3.16)
project(leadiel-ip VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)


find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(COPY assets DESTINATION ${CMAKE_BINARY_DIR})

set(PROJECT_SOURCES
    main.cpp
    mainwindow.cpp
    mainwindow.h
    #mainwindow.ui
)


qt_add_executable(leadiel-ip
    MANUAL_FINALIZATION
    ${PROJECT_SOURCES}
    topbar.h topbar.cpp


    pipeline.h pipeline.cpp
    sidebar.h sidebar.cpp
    viewer.h viewer.cpp
    my_line_edit.h
    functions.h
    functions.cpp
)

target_include_directories(leadiel-ip
    PRIVATE
        "C:/Sources/lime/sources"
)

if(MSVC)
    target_link_libraries(leadiel-ip PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
        debug ${OpenCV_DIR}/opencv_world4110d.lib
        optimized ${OpenCV_DIR}/opencv_world4110.lib
    )
else()
    target_link_libraries(leadiel-ip PRIVATE
        Qt${QT_VERSION_MAJOR}::Widgets
        ${OpenCV_LIBS}
    )
endif()

set_target_properties(leadiel-ip PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

include(GNUInstallDirs)
install(TARGETS leadiel-ip
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(leadiel-ip)
endif()
