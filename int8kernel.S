#ifdef __aarch64__

#define dst x0
#define dst_ptr_0 x9
#define dst_ptr_1 x10
#define dst_ptr_2 x11
#define dst_ptr_3 x12
#define tmp_ptr_a x13
#define tmp_ptr_b x14
.text
.align 5
.global int8kernel

int8kernel:
//void int8kernel(int32_t* dst, const uint8_t* src, const uint8_t* weight, size_t src_depth_quad, size_t dst_depth_quad);
//Auto: x0: dst, x1: src, x2:weight, x3: src_depth_quad
// x4: dst_depth_quad

.macro INIT
    eor v8.8b, v8.8b, v8.8b
    eor v9.8b, v9.8b, v9.8b
    eor v10.8b, v10.8b, v10.8b
    eor v11.8b, v11.8b, v11.8b
    eor v12.8b, v12.8b, v12.8b
    eor v13.8b, v13.8b, v13.8b
    eor v14.8b, v14.8b, v14.8b
    eor v15.8b, v15.8b, v15.8b

    eor v16.8b, v16.8b, v16.8b
    eor v17.8b, v17.8b, v17.8b
    eor v18.8b, v18.8b, v18.8b
    eor v19.8b, v19.8b, v19.8b
    eor v20.8b, v20.8b, v20.8b
    eor v21.8b, v21.8b, v21.8b
    eor v22.8b, v22.8b, v22.8b
    eor v23.8b, v23.8b, v23.8b
.endm

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64


lsl x8, x4, #4
mov dst_ptr_0, dst
add dst_ptr_1, dst_ptr_0, x8
add dst_ptr_2, dst_ptr_1, x8
add dst_ptr_3, dst_ptr_2, x8

L6LoopDz:
INIT
    mov x6, x1  // PanelA
#    subs x7, x3, #1
    mov x7, x3

    lsr x5, x3, #1
    lsl x15, x5, #1
    subs x15, x3, x15 // x15 = x3 % 2
    beq .L6LoopSzBegin 

    subs x7, x7, #1
    ld1 {v4.8b, v5.8b, v6.8b, v7.8b}, [x2], #32 // load four lines of B
    ld1 {v2.8b, v3.8b}, [x1], #16  // load two lines of PanelA
    smull v0.8h, v4.8b, v2.8b
    smull v1.8h, v4.8b, v3.8b

    saddlp v8.4s, v0.8h
    saddlp v12.4s, v1.8h
    smull v0.8h, v5.8b, v2.8b
    smull v1.8h, v5.8b, v3.8b
    saddlp v9.4s, v0.8h
    saddlp v13.4s, v1.8h
    smull v0.8h, v6.8b, v2.8b
    smull v1.8h, v6.8b, v3.8b
    saddlp v10.4s, v0.8h
    saddlp v14.4s, v1.8h
    smull v0.8h, v7.8b, v2.8b
    smull v1.8h, v7.8b, v3.8b
    saddlp v11.4s, v0.8h
    ld1 {v2.8b, v3.8b}, [x1], #16
    saddlp v15.4s, v1.8h
    smull v0.8h, v4.8b, v2.8b
    smull v1.8h, v4.8b, v3.8b
    saddlp v16.4s, v0.8h
    saddlp v20.4s, v1.8h
    smull v0.8h, v5.8b, v2.8b
    smull v1.8h, v5.8b, v3.8b
    saddlp v17.4s, v0.8h
    saddlp v21.4s, v1.8h
    smull v0.8h, v6.8b, v2.8b
    smull v1.8h, v6.8b, v3.8b
    saddlp v18.4s, v0.8h
    saddlp v22.4s, v1.8h
    smull v0.8h, v7.8b, v2.8b
    smull v1.8h, v7.8b, v3.8b
    saddlp v19.4s, v0.8h
    saddlp v23.4s, v1.8h

    cmp x7, #0
    beq L6LoopSzEnd

    .L6LoopSzBegin:
    L6LoopSz:
        add tmp_ptr_b, x2, #32 
        add tmp_ptr_a, x1, #32
        ld1 {v4.8b, v5.8b}, [x2], #16
        ld1 {v2.8b, v3.8b}, [x1], #16

        smull v0.8h, v4.8b, v2.8b
        ld1 {v6.8b, v7.8b}, [tmp_ptr_b], #16
        smull v1.8h, v5.8b, v2.8b
        ld1 {v24.8b, v25.8b}, [tmp_ptr_a], #16

        smlal v0.8h, v6.8b, v24.8b
        smlal v1.8h, v7.8b, v24.8b

        sadalp v8.4s, v0.8h
        sadalp v9.4s, v1.8h
    
        smull v0.8h, v4.8b, v3.8b
        smull v1.8h, v5.8b, v3.8b
        smlal v0.8h, v6.8b, v25.8b
        smlal v1.8h, v7.8b, v25.8b

        sadalp v12.4s, v0.8h
        sadalp v13.4s, v1.8h

        // finish v8v9 v12v13, start proc v16v17,v20v21
        ld1 {v28.8b, v29.8b}, [x1], #16
        smull v0.8h, v4.8b, v28.8b
        smull v1.8h, v5.8b, v28.8b
        ld1 {v26.8b, v27.8b}, [tmp_ptr_a], #16
        smlal v0.8h, v6.8b, v26.8b
        smlal v1.8h, v7.8b, v26.8b
        sadalp v16.4s, v0.8h
        sadalp v17.4s, v1.8h

        smull v0.8h, v4.8b, v29.8b
        smull v1.8h, v5.8b, v29.8b
        smlal v0.8h, v6.8b, v27.8b
        smlal v1.8h, v7.8b, v27.8b
        sadalp v20.4s, v0.8h
        sadalp v21.4s, v1.8h

        // start v10v11, v14v15, v18v19, v22v23, error here!
        ld1 {v4.8b, v5.8b}, [x2], #16
        smull v0.8h, v4.8b, v2.8b
        smull v1.8h, v5.8b, v2.8b
        ld1 {v6.8b, v7.8b}, [tmp_ptr_b], #16
        smlal v0.8h, v6.8b, v24.8b
        smlal v1.8h, v7.8b, v24.8b
        sadalp v10.4s, v0.8h
        sadalp v11.4s, v1.8h

        smull v0.8h, v4.8b, v3.8b
        smull v1.8h, v5.8b, v3.8b
        smlal v0.8h, v6.8b, v25.8b
        smlal v1.8h, v7.8b, v25.8b
        sadalp v14.4s, v0.8h
        sadalp v15.4s, v1.8h

        smull v0.8h, v4.8b, v28.8b
        smull v1.8h, v5.8b, v28.8b
        smlal v0.8h, v6.8b, v26.8b
        smlal v1.8h, v7.8b, v26.8b
        sadalp v18.4s, v0.8h
        sadalp v19.4s, v1.8h

        smull v0.8h, v4.8b, v29.8b
        smull v1.8h, v5.8b, v29.8b
        smlal v0.8h, v6.8b, v27.8b
        smlal v1.8h, v7.8b, v27.8b
        sadalp v22.4s, v0.8h
        sadalp v23.4s, v1.8h

        add x1, x1, #32
        add x2, x2, #32
        subs x7, x7, #2
        bne L6LoopSz
    L6LoopSzEnd:

    addp v8.4s, v8.4s, v9.4s
    addp v10.4s, v10.4s, v11.4s
    addp v12.4s, v12.4s, v13.4s
    addp v14.4s, v14.4s, v15.4s
    addp v16.4s, v16.4s, v17.4s
    addp v18.4s, v18.4s, v19.4s
    addp v20.4s, v20.4s, v21.4s
    addp v22.4s, v22.4s, v23.4s

    addp v8.4s, v8.4s, v10.4s
    addp v9.4s, v12.4s, v14.4s
    addp v10.4s, v16.4s, v18.4s
    addp v11.4s, v20.4s, v22.4s

    st1 {v8.4s}, [dst_ptr_0], #16
    st1 {v9.4s}, [dst_ptr_1], #16
    st1 {v10.4s}, [dst_ptr_2], #16
    st1 {v11.4s}, [dst_ptr_3], #16

    subs x4, x4, #1
    mov x1, x6
    bne L6LoopDz

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
