#ifdef __aarch64__

#define dst         x0
#define src_ori     x6
#define kd8_ori     x7
#define dst_ptr_0   x9
#define dst_ptr_1   x10
#define dst_ptr_2   x11
#define dst_ptr_3   x12
#define tmp_ptr_a   x13
#define tmp_ptr_b   x14

#define is_kd8_even x15
#define kd4_ori     x16
#define kd2_ori     x17
#define kd1_ori     x18

.text
.align 5
.global int8kernel_m4

int8kernel_m4:
//void int8kernel_m4(int32_t* dst, const uint8_t* src, const uint8_t* weight, size_t k, size_t n);
//x0: dst, x1: src, x2:weight, x3: k, x4: n

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

.macro DECOMPOSE_K
    // divide k into  8*kd8_ori + 4*kd4_ori + 2*kd2_ori + 1*kd1_ori
    lsr kd8_ori, x3, 3  // kd8_ori = x3 / 8, means loopk
    lsr x5, kd8_ori, 1
    lsl is_kd8_even, x5, 1
    sub is_kd8_even, kd8_ori, is_kd8_even // x15 = x3 % 2
    
    lsr x5, kd8_ori, 3
    sub x3, x3, x5
    lsr kd4_ori, x3, 2  
    lsl x5, kd4_ori, 2
    sub x3, x3, x5
    
    lsr kd2_ori, x3, 1
    lsl x5, kd2_ori, 1
    sub x3, x3, x5
    
    mov kd1_ori, x3
.endm

sub sp, sp, #128
st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64

lsl x8, x4, #2
mov dst_ptr_0, dst
add dst_ptr_1, dst_ptr_0, x8
add dst_ptr_2, dst_ptr_1, x8
add dst_ptr_3, dst_ptr_2, x8

DECOMPOSE_K

loopnd4:
    INIT
    mov src_ori, x1  // PanelA
    
    cmp is_kd8_even, 0 
    beq loopkd8_nd4_even // loop number is even 

    // start loopkd8_nd4
    subs kd8_ori, kd8_ori, #1
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

    cmp kd8_ori, #0
    beq loopkd8_nd4_end 

    loopkd8_nd4_even:
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
        subs kd8_ori, kd8_ori, #2
        bne loopkd8_nd4_even

    loopkd8_nd4_end:
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

        // start process kd4 kd2 kd1 cases
        cmp kd4_ori, 0
        beq loopkd2_nd4 
            // start subkernel_m4n4k4
            ld1 {v4.8b, v5.8b}, [x2], #16  // load B4x4
            sxtl v4.8h, v4.8b
            sxtl v5.8h, v5.8b
            mov v6.d[0], v4.d[1]
            mov v7.d[0], v5.d[1]

            ld1 {v2.8b}, [x1], #8  // load A2x4
            sxtl v2.8h, v2.8b
            mov v3.d[0], v2.d[1]

            smull v12.4s, v2.4h, v4.4h
            smull v13.4s, v2.4h, v5.4h
            smull v14.4s, v2.4h, v6.4h
            smull v15.4s, v2.4h, v7.4h

            addp v12.4s, v12.4s, v13.4s
            addp v14.4s, v14.4s, v15.4s
            addp v12.4s, v12.4s, v14.4s
            add v8.4s, v8.4s, v12.4s

            smull v16.4s, v3.4h, v4.4h
            smull v17.4s, v3.4h, v5.4h
            smull v18.4s, v3.4h, v6.4h
            smull v19.4s, v3.4h, v7.4h

            addp v16.4s, v16.4s, v17.4s
            addp v18.4s, v18.4s, v19.4s
            addp v16.4s, v16.4s, v18.4s
            add v9.4s, v9.4s, v16.4s

            ld {v2.8b}, [x1], #8 // loadd A2x4
            sxtl v2.8h, v2.8b
            mov v3.d[0], v2.d[1]

            smull v12.4s, v2.4h, v4.4h
            smull v13.4s, v2.4h, v5.4h
            smull v14.4s, v2.4h, v6.4h
            smull v15.4s, v2.4h, v7.4h

            addp v12.4s, v12.4s, v13.4s
            addp v14.4s, v14.4s, v15.4s
            addp v12.4s, v12.4s, v14.4s
            add v10.4s, v10.4s, v12.4s

            smull v16.4s, v3.4h, v4.4h
            smull v17.4s, v3.4h, v5.4h
            smull v18.4s, v3.4h, v6.4h
            smull v19.4s, v3.4h, v7.4h

            addp v16.4s, v16.4s, v17.4s
            addp v18.4s, v18.4s, v19.4s
            addp v16.4s, v16.4s, v18.4s
            add v11.4s, v11.4s, v16.4s

        loopkd2_nd4:
            cmp kd2_ori, 0
            beq loopkd1_nd4 
            // start subkernel_m4n4k2, TODO butterfly

        loopkd1_nd4:
            cmp kd1_ori, 0 
            beq loopnd4_end
            // start subkernel_m4n4k1
            ld1 {v4.8b}, [x2], #4   // load B1x4
            ld1 {v2.8b}, [x1], #4   // load A4x1
            sxtl v4.8h, v4.8b
            sxtl v2.8h, v2.8b
            smlal v8.4s, v4.4h, v2.h[0]
            smlal v9.4s, v4.4h, v2.h[1]
            smlal v10.4s, v4.4h, v2.h[2]
            smlal v11.4s, v4.4h, v2.h[3]

    loopnd4_end:
        st1 {v8.4s}, [dst_ptr_0], #16
        st1 {v9.4s}, [dst_ptr_1], #16
        st1 {v10.4s}, [dst_ptr_2], #16
        st1 {v11.4s}, [dst_ptr_3], #16

        subs x4, x4, #1
        mov x1, src_ori 
        bne loop_nd4 

sub sp, sp, #128
ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
ret

#endif
