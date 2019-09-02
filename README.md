### symmint8 gemm


chgemm is a int8 gemm project.
Just like gemmlowp Nonlocal, input value cannot be -128.

chgemm 是一个 int8 gemm 工程，与 BLAS gemm 不完全相同：

1. 更多地为深度学习应用场景考虑，将提供为 2D 卷积优化适配的接口
2. 更考虑伸手党，将直接提供各平台编译好的 exe 和 benchmark 结果
3. 专注 gemm 一百年，不低于其他项目的 symmint8 gemm 速度（反正先把 flag 立好再说 ^_^）。
