### chgemm

chgemm is an symmetric int8 project, which is slightly different from BLAS sgemm:
1. when you input an int8_t type of matrix [-127,+127], you will get an int32_t one. PS: pay attention to the overflow;
2. considering the application scene of the deeep learning, the packAB interface is open and can be adjusted;
3. the common design plan is `alpha*A*B+beta*C=C`, but mine is `C=A*B`, because they have no utility in deep learning inference;
4. the speed of this project is not slower than any other projects' symmint8 gemm.

chgemm 是一个 int8 gemm 工程，与 BLAS gemm 不完全相同：

1. 输入为 [-127, +127] 范围内的 int8_t 类型矩阵，输出为 int32_t 矩阵。需注意溢出；
2. 更多地为深度学习应用场景考虑，packAB 接口暴露出来可以调整；
3. 实现为 C = A * B。alpha 和 beta 在深度学习推理中无实用意义；
3. 不低于其他项目的 symmint8 gemm 速度。

### test result
Compiled on RK3399 with `-O3` flag. The current peek can be 18.6 gflops, and the orange line is the single-core fp32 limit(14.3 gflops). 

### 速度
-O3 编译，目前在 rk3399 单核结果。目前极限可以到 18.6 gflops，橙线是 rk3399 单核 fp32 极限。 

![尺寸和gflops结果](0.png)
