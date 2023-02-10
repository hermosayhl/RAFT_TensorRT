-- 设置工程名
set_project("raft_estimator")
-- 设置工程版本
set_version("0.0.1")
-- 设置 xmake 版本
set_xmakever("2.1.0")
-- 设置支持的平台
set_allowedplats("windows", "linux")



-- 编译共有的一些文件, 避免每个目标都重新编译
target("raft_infer")
    -- 生成类型, 共享库
    set_kind("binary")
    -- 设置 C/C++ 标准
    set_languages("c99", "cxx17")
    -- 开启警告
    set_warnings("all")
    -- 设置 C/C++ 标准
    set_languages("c99", "cxx17")
    -- 开启优化
    if is_mode("release") then
        set_optimize("fastest")
        print("release")
    end

    -- 设置第三方库
    --  1. 添加 OpenCV
    if is_os("windows") then
        opencv_root    = "D:/environments/C++/OpenCV/opencv-msvc/build"
        opencv_version = "452"
        add_linkdirs(opencv_root .. "/x64/vc15/lib/")
        add_links(
            "opencv_world" .. opencv_version
        )
        add_includedirs(opencv_root .. "/include")
    end
    if is_os("linux") then
        opencv_root    = "/home/dx/usrs/liuchang/tools/opencv/build/install"
        add_linkdirs(opencv_root .. "/lib")
        add_links(
            "opencv_core",
            "opencv_highgui",
            "opencv_imgcodecs",
            "opencv_imgproc"
        )
        add_includedirs(opencv_root .. "/include/opencv4")
    end
    
    -- 2. 添加 CUDA
    cuda_root = "/usr/local/cuda"
    add_includedirs(cuda_root .. "/include")
    if is_os("windows") then
        add_linkdirs(cuda_root .. "/lib/x64")
    end
    if is_os("linux") then
        add_linkdirs(cuda_root .. "/lib64")
    end
    add_links(
        "cudart"
    )

    -- 3. 添加 TensorRT
    local tensorrt_root = "/home/dx/usrs/liuchang/tools/TensorRT-8.5.3.1"
    add_includedirs(tensorrt_root .. "/include")
    add_includedirs(tensorrt_root .. "/samples/common")
    if is_os("windows") then
        add_linkdirs(tensorrt_root .. "/lib")
    end
    if is_os("linux") then
        add_linkdirs(tensorrt_root .. "/targets/x86_64-linux-gnu/lib")
    end
    add_links(
        "nvinfer",
        "nvonnxparser",
        "nvinfer_plugin"
    )

    -- 添加自己的项目源文件
    add_files(
        "$(projectdir)/src/inference.cpp", 
        tensorrt_root .. "/samples/common/logger.cpp"
    )
    -- 添加 include 自己的代码
    add_includedirs("$(projectdir)/include/")

    -- 设置目标工作目录
    set_rundir("$(projectdir)")

target_end()


