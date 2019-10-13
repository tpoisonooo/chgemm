#ifdef __aarch64__

#define  ptr_src  x0
#define  ptr_dst0 x1
#define  k        x2
#define  n        x3
#define  ptr0_0   x4
#define  ptr0_1   x5
#define  ptr0_2   x6
#define  ptr0_3   x7
#define  ptr0_4   x8
#define  ptr0_5   x9
#define  ptr0_6   x10
#define  ptr0_7   x11
#define  nd8      x12
#define  nl       x13
#define  kd       x14
#define  kd8_ori  x15
#define  kd4_ori  x16
#define  kd2_ori  x17
#define  kd1_ori  x18
#define  temp1    w20
#define  temp2    w21
#define  k_temp   x22
#define  ptr_dst1 x23
#define  step     x24

.text
.align 5
.global reorder_b
.type reorder_b, %function

.macro INIT_LOOP_PARAMETERS
    lsr nd8, n, #3
    lsl nd8, nd8, #3
    subs nl, n, nd8
    lsr nd8, n, #3

    // divide height into 8x + 4y + 2z + 1w, xyzw is natural number.
    mov k_temp, k
    lsr kd8_ori, k, 3
    lsl kd, kd8_ori, 3
    sub k_temp, k_temp, kd

    lsr kd4_ori, k_temp, 2
    lsl kd, kd4_ori, 2
    subs k_temp, k_temp, kd

    lsr kd2_ori, k_temp, 1
    lsl kd, kd2_ori, 1
    subs k_temp, k_temp, kd

    mov kd1_ori, k_temp
.endm
    
    
reorder_b:
//void reorder_b(int8_t* src, int8_t* dst, size_t k, size_t n);

    add sp, sp, #-(11 * 16)
    stp d8, d9, [sp, #(0 * 16)]
    stp d10, d11, [sp, #(1 * 16)]
    stp d12, d13, [sp, #(2 * 16)]
    stp d14, d15, [sp, #(3 * 16)]
    stp d16, d17, [sp, #(4 * 16)]
    stp x18, x19, [sp, #(5 * 16)]
    stp x20, x21, [sp, #(6 * 16)]
    stp x22, x23, [sp, #(7 * 16)]
    stp x24, x25, [sp, #(8 * 16)]
    stp x26, x27, [sp, #(9 * 16)]
    str x28, [sp, #(10 * 16)]
    
    INIT_LOOP_PARAMETERS 
    
    mov ptr_dst0, x1
    
// divide height into 4x + 2y + 1z, xyz is natural number.
    mov ptr0_0, ptr_src
    add ptr0_1, ptr0_0, n 
    add ptr0_2, ptr0_1, n 
    add ptr0_3, ptr0_2, n 
    add ptr0_4, ptr0_3, n 
    add ptr0_5, ptr0_4, n 
    add ptr0_6, ptr0_5, n 
    add ptr0_7, ptr0_6, n 

    cmp nd8, #0
    beq loopnd_ln

loopnd8:
    lsl step, k, #2
    add ptr_dst1, ptr_dst0, step

    // if height < 8, skip.
    cmp kd8_ori, 0
    beq loopnd8_kd4

	mov kd, kd8_ori
    lsl step, n, #3
    loopnd8_kd8:
        ld1 {v0.8b}, [ptr0_0], step 
        ld1 {v1.8b}, [ptr0_1], step 
        trn1 v8.8b, v0.8b, v1.8b
        ld1 {v2.8b}, [ptr0_2], step 
        trn2 v9.8b, v0.8b, v1.8b
        ld1 {v3.8b}, [ptr0_3], step 
        trn1 v0.4h, v8.4h, v9.4h
        ld1 {v4.8b}, [ptr0_4], step 
        trn2 v1.4h, v8.4h, v9.4h
        ld1 {v5.8b}, [ptr0_5], step 
        trn1 v8.2s, v0.2s, v1.2s
        ld1 {v6.8b}, [ptr0_6], step 
        trn2 v16.2s, v0.2s, v1.2s
        ld1 {v7.8b}, [ptr0_7], step 


        trn1 v10.8b, v2.8b, v3.8b
        trn2 v11.8b, v2.8b, v3.8b
        trn1 v2.4h, v10.4h, v11.4h
        trn2 v3.4h, v10.4h, v11.4h
        trn1 v9.2s, v2.2s, v3.2s
        trn2 v17.2s, v2.2s, v3.2s

        trn1 v12.8b, v4.8b, v5.8b
        trn2 v13.8b, v4.8b, v5.8b
        trn1 v4.4h, v12.4h, v13.4h
        trn2 v5.4h, v12.4h, v13.4h
        trn1 v10.2s, v4.2s, v5.2s
        trn2 v18.2s, v4.2s, v5.2s

        trn1 v14.8b, v6.8b, v7.8b
        trn2 v15.8b, v6.8b, v7.8b
        trn1 v6.4h, v14.4h, v15.4h
        trn2 v7.4h, v14.4h, v15.4h
        trn1 v11.2s, v6.2s, v7.2s
        st4 {v8.4h, v9.4h, v10.4h, v11.4h}, [ptr_dst0], #32
        trn2 v19.2s, v6.2s, v7.2s
        st4 {v16.4h, v17.4h, v18.4h, v19.4h}, [ptr_dst1], #32

        subs kd, kd, #1
        bne loopnd8_kd8
    
    loopnd8_kd4:
        cmp kd4_ori, 0
        beq loopnd8_kd2
        lsl step, n, #2
        ld1 {v0.8b}, [ptr0_0], step
        ld1 {v1.8b}, [ptr0_1], step
        trn1 v4.8b, v0.8b, v1.8b
        ld1 {v2.8b}, [ptr0_2], step
        trn2 v5.8b, v0.8b, v1.8b
        ld1 {v3.8b}, [ptr0_3], step

        trn1 v6.8b, v2.8b, v3.8b
        trn2 v7.8b, v2.8b, v3.8b

        trn1 v0.4h, v4.4h, v6.4h
        trn2 v1.4h, v4.4h, v6.4h
        trn1 v2.4h, v5.4h, v7.4h
        trn2 v3.4h, v5.4h, v7.4h

        mov v8.s[0], v0.s[1]
        mov v8.s[1], v2.s[1]
        mov v9.s[0], v1.s[1]
        mov v9.s[1], v3.s[1]
        st1 {v8.8b, v9.8b}, [ptr_dst1], #16

        mov v0.s[1], v2.s[0]
        mov v1.s[1], v3.s[0]
        st1 {v0.8b, v1.8b}, [ptr_dst0], #16
        
    loopnd8_kd2:
        cmp kd2_ori, 0
        beq loopnd8_kd1
        lsl step, n, #1

        ld1 {v0.8b}, [ptr0_0], step
        ld1 {v1.8b}, [ptr0_1], step

        trn1 v2.8b, v0.8b, v1.8b
        trn2 v3.8b, v0.8b, v1.8b
        trn1 v0.4h, v2.4h, v3.4h
        trn2 v1.4h, v2.4h, v3.4h
        trn1 v2.2s, v0.2s, v1.2s

        st1 {v2.8b}, [ptr_dst0], #8
        trn2 v3.2s, v0.2s, v1.2s
        st1 {v3.8b}, [ptr_dst1], #8
        
    loopnd8_kd1:
        cmp kd1_ori, 0
        beq loopnd8_end

        ldrb temp1, [ptr0_0], #1
        strb temp1, [ptr_dst0], #1
        ldrb temp2, [ptr0_0], #1
        strb temp2, [ptr_dst0], #1
        ldrb temp1, [ptr0_0], #1
        strb temp1, [ptr_dst0], #1
        ldrb temp2, [ptr0_0], #1
        strb temp2, [ptr_dst0], #1

        ldrb temp1, [ptr0_0], #1
        strb temp1, [ptr_dst1], #1
        ldrb temp2, [ptr0_0], #1
        strb temp2, [ptr_dst1], #1
        ldrb temp1, [ptr0_0], #1
        strb temp1, [ptr_dst1], #1
        ldrb temp2, [ptr0_0], #1
        strb temp2, [ptr_dst1], #1

    loopnd8_end:
        mov ptr_dst0, ptr_dst1

        add ptr_src, ptr_src, #8 
        mov ptr0_0, ptr_src
        add ptr0_1, ptr0_0, n 
        add ptr0_2, ptr0_1, n 
        add ptr0_3, ptr0_2, n 
        add ptr0_4, ptr0_3, n 
        add ptr0_5, ptr0_4, n 
        add ptr0_6, ptr0_5, n 
        add ptr0_7, ptr0_6, n 

        subs nd8, nd8, #1
        bne loopnd8
        
loopnd_ln:
    cmp nl, #0
    beq end
    cmp nl, #4
    bge loopnd4_start
    cmp nl, #2
    bge loopnd2_start
    b loopnd1_start

loopnd4_start:
    sub nl, nl, #4

    cmp kd8_ori, #0
    beq loopnd4_kd4

	mov kd, kd8_ori
    lsl step, n, #3
    loopnd4_kd8:
        ld1 {v0.8b}, [ptr0_0], step 
        ld1 {v1.8b}, [ptr0_1], step 
        trn1 v8.8b, v0.8b, v1.8b
        ld1 {v2.8b}, [ptr0_2], step 
        trn2 v9.8b, v0.8b, v1.8b
        ld1 {v3.8b}, [ptr0_3], step 
        trn1 v16.4h, v8.4h, v9.4h
        ld1 {v4.8b}, [ptr0_4], step 
        trn2 v20.4h, v8.4h, v9.4h
        ld1 {v5.8b}, [ptr0_5], step 
        trn1 v10.8b, v2.8b, v3.8b
        ld1 {v6.8b}, [ptr0_6], step 
        trn2 v11.8b, v2.8b, v3.8b
        ld1 {v7.8b}, [ptr0_7], step 
         
        trn1 v17.4h, v10.4h, v11.4h
        trn2 v21.4h, v10.4h, v11.4h

        trn1 v12.8b, v4.8b, v5.8b
        trn2 v13.8b, v4.8b, v5.8b
        trn1 v18.4h, v12.4h, v13.4h
        trn2 v22.4h, v12.4h, v13.4h

        trn1 v14.8b, v6.8b, v7.8b
        trn2 v15.8b, v6.8b, v7.8b
        trn1 v19.4h, v14.4h, v15.4h
        trn2 v23.4h, v14.4h, v15.4h

        st4 {v16.4h, v17.4h, v18.4h, v19.4h}, [ptr_dst0]
        add ptr_dst0, ptr_dst0, #16
        st4 {v20.4h, v21.4h, v22.4h, v23.4h}, [ptr_dst0]
        add ptr_dst0, ptr_dst0, #16

        subs kd, kd, #1
        bne loopnd4_kd8

    loopnd4_kd4:
        cmp kd4_ori, #0
        beq loopnd4_kd2

        lsl step, n, #2
        ld1 {v0.8b}, [ptr0_0], step
        ld1 {v1.8b}, [ptr0_1], step
        trn1 v4.8b, v0.8b, v1.8b
        ld1 {v2.8b}, [ptr0_2], step
        trn2 v5.8b, v0.8b, v1.8b
        ld1 {v3.8b}, [ptr0_3], step
        
        trn1 v6.8b, v2.8b, v3.8b
        trn2 v7.8b, v2.8b, v3.8b
        
        trn1 v8.4h, v4.4h, v6.4h
        trn2 v9.4h, v4.4h, v6.4h
        trn1 v10.4h, v5.4h, v7.4h
        trn2 v11.4h, v5.4h, v7.4h

        mov v8.s[1], v10.s[0]
        mov v9.s[1], v11.s[0]

        st1 {v8.8b, v9.8b}, [ptr_dst0], #16
        
    loopnd4_kd2:
        cmp kd2_ori, #0
        beq loopnd4_kd1

        lsl step, n, #1
        ld1 {v0.8b}, [ptr0_0], step
        ld1 {v1.8b}, [ptr0_1], step
        st2 {v0.8b, v1.8b}, [ptr_dst0]
        add ptr_dst0, ptr_dst0, #8

    loopnd4_kd1:
        cmp kd1_ori, #0
        beq loopnd4_end
    
        ld1 {v0.8b}, [ptr0_0], n
        st1 {v0.8b}, [ptr_dst0]
        add ptr_dst0, ptr_dst0, #4

    loopnd4_end:
        add ptr_src, ptr_src, #4 
        mov ptr0_0, ptr_src
        add ptr0_1, ptr0_0, n 
        add ptr0_2, ptr0_1, n 
        add ptr0_3, ptr0_2, n 
        add ptr0_4, ptr0_3, n 
        add ptr0_5, ptr0_4, n 
        add ptr0_6, ptr0_5, n 
        add ptr0_7, ptr0_6, n 
        b loopnd_ln

loopnd2_start:
// proc left two columns.
    sub nl, nl, #2
    cmp kd8_ori, 0
    beq loopnd2_kd4

	mov kd, kd8_ori
    lsl step, n, #3
    loopnd2_kd8:
        ld1 {v0.8b}, [ptr0_0], step 
        ld1 {v1.8b}, [ptr0_1], step 
        ld1 {v2.8b}, [ptr0_2], step 
        ld1 {v3.8b}, [ptr0_3], step 
        mov v0.h[1], v2.h[0]
        ld1 {v4.8b}, [ptr0_4], step 
        ld1 {v5.8b}, [ptr0_5], step 
        mov v0.h[2], v4.h[0]
        ld1 {v6.8b}, [ptr0_6], step 
        ld1 {v7.8b}, [ptr0_7], step 
        mov v0.h[3], v6.h[0]
         
        mov v1.h[1], v3.h[0]
        mov v1.h[2], v5.h[0]
        mov v1.h[3], v7.h[0]

        trn1 v2.8b, v0.8b, v1.8b
        trn2 v3.8b, v0.8b, v1.8b

        st1 {v2.8b, v3.8b}, [ptr_dst0], #16

        subs kd, kd, #1
        bne loopnd2_kd8

    loopnd2_kd4:
        cmp kd4_ori, #0
        beq loopnd2_kd2

        lsl step, n, #2
        ld1 {v0.8b}, [ptr0_0], step
        ld1 {v1.8b}, [ptr0_1], step
        ld1 {v2.8b}, [ptr0_2], step
        ld1 {v3.8b}, [ptr0_3], step

        mov v0.h[1], v2.h[0]
        mov v1.h[1], v3.h[0]

        trn1 v2.8b, v0.8b, v1.8b
        trn2 v3.8b, v0.8b, v1.8b
        mov v2.s[1], v3.s[0]

        st1 {v2.8b}, [ptr_dst0], #8
        
    loopnd2_kd2:
        cmp kd2_ori, #0
        beq loopnd2_kd1

        lsl step, n, #1
        sub step, step, #2

        ldrb temp1, [ptr0_0], #1
        strb temp1, [ptr_dst0], #1

        ldrb temp2, [ptr0_0], #1
        ldrb temp1, [ptr0_1], #1
        strb temp1, [ptr_dst0], #1
        strb temp2, [ptr_dst0], #1 
        
        ldrb temp1, [ptr0_1], #1
        strb temp1, [ptr_dst0], #1

        add ptr0_0, ptr0_0, step
        add ptr0_1, ptr0_1, step
        
    loopnd2_kd1:
        cmp kd1_ori, #0
        beq loopnd2_end
    
        add k_temp, ptr0_0, n
        ldrb temp1, [ptr0_0], #1
        strb temp1, [ptr_dst0], #1
        ldrb temp2, [ptr0_0], #1
        strb temp2, [ptr_dst0], #1
        mov ptr0_0, k_temp

    loopnd2_end:
        add ptr_src, ptr_src, #2 
        mov ptr0_0, ptr_src
        b loopnd_ln

// proc the last line
loopnd1_start:
    sub nl, nl, #1

    copy:
        ldrb temp1, [ptr0_0]
        add ptr0_0, ptr0_0, n
        strb temp1, [ptr_dst0], #1
        subs k, k, #1
        bne copy 
   
end:
    mov x0, #0              // set return value
    ldp d8, d9, [sp, #(0 * 16)]
    ldp d10, d11, [sp, #(1 * 16)]
    ldp d12, d13, [sp, #(2 * 16)]
    ldp d14, d15, [sp, #(3 * 16)]
    ldp d16, d17, [sp, #(4 * 16)]
    ldp x18, x19, [sp, #(5 * 16)]
    ldp x20, x21, [sp, #(6 * 16)]
    ldp x22, x23, [sp, #(7 * 16)]
    ldp x24, x25, [sp, #(8 * 16)]
    ldp x26, x27, [sp, #(9 * 16)]
    ldr x28, [sp, #(10 * 16)]
    add sp, sp, #(11*16)
    ret
#endif
