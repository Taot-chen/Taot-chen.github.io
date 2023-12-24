---
layout: post
title: python_memory_profiler
date: 2023-12-23
---


## memory_profiler python 代码内存性能分析

* 安装

    ```bash
    pip install memory_profiler
    ```

* 使用

    ```python
    # 在需要分析内存性能的函数前面添加修饰符 @profiler
    @profiler
    @profiler(precision = 4, stream=open("memory.info", "w+"))	# 配置精度，并且把结果输出到日志文件
    def aaa()；
    	a = []
        b = []
        c = []
        for i in range(10):
            a.append(i)
    ```

    * 运行

        ```bash
        python -m memory_profiler test.py
        ```

    * 可视化工具 mprof

        ```bash
        python -m mprof run test.py		# 生成 memory perf 数据
        python -m mprof plot memory.info		# 绘制 memory perf 图，并显示
        python -m mprof plot memory.info --output=memory.png		# 绘制 memory perf 图，不显示，保存至当前路径
        ```

    * mprof 命令

        ```bash
        mprof run，运行可执行文件，记录内存使用情况
        mprof plot，绘制一个记录的内存使用情况，默认情况下是最后一个
        mprof list，以用户友好的方式列出所有记录的内存使用情况文件
        mprof clean，删除所有记录的内存使用情况文件
        mprof rm，删除特定记录的内存使用情况文件
        ```

    * 跟踪子进程

        ```bash
        方式一，总结所有子进程的内存和父进程的内存使用情况并跟踪每个子进程
        mprof run --include-children <script>
        
        方式二，独立于主进程跟踪每个子进程，通过索引将子进程内存消耗情况序列化到输出流，使用多进程
        mprof run --multiprocess <scripts>
        ```

    * 根据内存使用量设置断点

        ```bash
        python -m memory_profiler --pdb-mem=100 test.py		# 一旦代码在装饰函数中使用超过 100MB 的内存，将中断 test.py 并进入 pdb 调试器
        ```

* python 内存优化思路

    * 使用到的临时变量，即使释放，`delete variable`
    * 考虑到 python 特殊的内存管理机制，可以把一项工作按照内存消耗拆分成一系列的单独的步骤，每个步骤放置在一个子进程中，所有的子进程串行，减少代码的内存消耗峰值



