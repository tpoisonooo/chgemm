### chgemm

chgemm is a symmetric int8 gemm project.
Just like gemmlowp Nonlocal, input value cannot be -128.

chgemm 是一个 int8 gemm 工程，与 BLAS gemm 不完全相同：

1. 更多地为深度学习应用场景考虑，将提供为 2D 卷积优化适配的接口
2. 更考虑伸手党，将直接提供各平台编译好的 exe 和 benchmark 结果
3. 专注 gemm 一百年，不低于其他项目的 symmint8 gemm 速度. 目前可达 16-17gflops. 但是有个小 Bug，必须 -O0 编译...正在处理。

### 速度
-O0 编译，目前结果
>
--------------------------------
M=123 N=123 K=123 REPEAT=10
OpenBLAS sgemm timecost:1.59057ms
timer cost for chgemm:1.25515ms
errors : 0
--------------------------------
M=321 N=321 K=321 REPEAT=10
OpenBLAS sgemm timecost:9.25144ms
timer cost for chgemm:5.70249ms
errors : 0
--------------------------------
M=1321 N=1321 K=1321 REPEAT=10
OpenBLAS sgemm timecost:406.735ms
timer cost for chgemm:278.287ms
errors : 0
--------------------------------
M=121 N=1321 K=1321 REPEAT=10
OpenBLAS sgemm timecost:41.2142ms
timer cost for chgemm:47.8499ms
errors : 0
--------------------------------
M=1321 N=121 K=1321 REPEAT=10
OpenBLAS sgemm timecost:40.6452ms
timer cost for chgemm:26.0947ms
errors : 0
--------------------------------
M=1321 N=1321 K=121 REPEAT=10
OpenBLAS sgemm timecost:43.6819ms
timer cost for chgemm:26.4824ms
errors : 0
--------------------------------
M=2000 N=2000 K=2000 REPEAT=10
OpenBLAS sgemm timecost:1329.04ms
timer cost for chgemm:887.612ms
errors : 0
