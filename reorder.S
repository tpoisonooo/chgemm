#ifdef __aarch64__

#define  ptr_src  x0
#define  ptr_dst  x1
#define  m        x2
#define  k        x3
#define  ptr0_0   x4
#define  ptr0_1   x5
#define  ptr0_2   x6
#define  ptr0_3   x7
#define  md4      x8
#define  ml       x9
#define  kd       x11
#define  kd8_ori  x12
#define  kd4_ori  x13
#define  kd2_ori  x14
#define  kd1_ori  x15
#define  kd8_l    x16
#define  temp1    w17
#define  temp2    w18
#define  k_temp   x19

.text
.align 5

.global reorder
reorder:
//void reorder(uint8_t* src, uint8_t* dst, size_t m, size_t k);
//Auto: x0: src, x1: dst, x2:m, x3:k

sub sp, sp, #64
st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [sp], #64

lsr md4, m, 2
lsl md4, md4, #2
subs ml, m, md4
lsr md4, m, 2

// divide width into 8x + 4y + 2z + 1w, xyzw is natural number.
lsr kd8_ori, k, 3
lsl kd, kd8_ori, 3
mov k_temp, k
subs k_temp, k_temp, kd
mov kd8_l, k_temp

lsr kd4_ori, k_temp, 2
lsl kd, kd4_ori, 2
subs k_temp, k_temp, kd
lsr kd2_ori, k_temp, 1
lsl kd, kd2_ori, 1
subs k_temp, k_temp, kd
mov kd1_ori, k_temp

mov ptr_dst, x1
mov ptr0_0, ptr_src
add ptr0_1, ptr0_0, k
add ptr0_2, ptr0_1, k
add ptr0_3, ptr0_2, k
subs md4, md4, #0
beq loopmd_ln

// divide height into 4x + 2y + 1z, xyz is natural number.
loopmd4:
    // if width < 8, skip.
    cmp kd8_ori, 0
    beq loopmd4_kd4

	mov kd, kd8_ori
    loopmd4_kd8:
        ld1 {v0.8b}, [ptr0_0], #8
        ld1 {v1.8b}, [ptr0_1], #8
        ld1 {v2.8b}, [ptr0_2], #8
        ld1 {v3.8b}, [ptr0_3], #8
        st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [ptr_dst], #32
        subs kd, kd, #1
        bne loopmd4_kd8
    
    loopmd4_kd4:
        cmp kd4_ori, 0
        beq loopmd4_kd2
        ld1 {v0.8b}, [ptr0_0]
        add ptr0_0, ptr0_0, 4
        ld1 {v1.8b}, [ptr0_1]
        add ptr0_1, ptr0_1, 4
        ld1 {v2.8b}, [ptr0_2]
        add ptr0_2, ptr0_2, 4
        ld1 {v3.8b}, [ptr0_3]
        add ptr0_3, ptr0_3, 4
    
        trn1 v0.2s, v0.2s, v1.2s
        st1 {v0.8b}, [ptr_dst], #8
        trn1 v2.2s, v2.2s, v3.2s
        st1 {v2.8b}, [ptr_dst], #8
    
    loopmd4_kd2:
        cmp kd2_ori, 0
        beq loopmd4_kd1
        ld1 {v0.8b}, [ptr0_0]
        add ptr0_0, ptr0_0, #2
        ld1 {v1.8b}, [ptr0_1]
        add ptr0_1, ptr0_1, #2
        ld1 {v2.8b}, [ptr0_2]
        add ptr0_2, ptr0_2, #2
        ld1 {v3.8b}, [ptr0_3]
        add ptr0_3, ptr0_3, #2
    
        trn1 v0.4h, v0.4h, v1.4h
        trn1 v2.4h, v2.4h, v3.4h
        trn1 v0.2s, v0.2s, v2.2s
        st1 {v0.8b}, [ptr_dst], #8
    
    loopmd4_kd1:
        cmp kd1_ori, 0
        beq loopmd4_continue

        ldrb temp1, [ptr0_0], #1
        strb temp1, [ptr_dst], #1
        ldrb temp2, [ptr0_1], #1
        strb temp2, [ptr_dst], #1
        ldrb temp1, [ptr0_2], #1
        strb temp1, [ptr_dst], #1
        ldrb temp2, [ptr0_3], #1
        strb temp2, [ptr_dst], #1

    loopmd4_continue:
        mov ptr0_0, ptr0_3
        add ptr0_1, ptr0_0, k
        add ptr0_2, ptr0_1, k
        add ptr0_3, ptr0_2, k
        subs md4, md4, #1
        bne loopmd4

loopmd_ln:
    cmp ml, #0
    beq end
    cmp ml, #2
    bge loopmd2_start
    b loopmd1_start

loopmd2_start:
// proc left two lines.
    sub ml, ml, #2
    cmp kd8_ori, 0
    beq loopmd2_kd4

    mov kd, kd8_ori
    loopmd2_kd8:
	    ld1 {v0.8b}, [ptr0_0], #8
	    ld1 {v1.8b}, [ptr0_1], #8
	    st1 {v0.8b, v1.8b}, [ptr_dst], #16
	    subs kd, kd, #1
	    bne loopmd2_kd8

    loopmd2_kd4:
        cmp kd4_ori, 0
        beq loopmd2_kd2
    
        ld1 {v0.8b}, [ptr0_0]
        add ptr0_0, ptr0_0, #4
        ld1 {v1.8b}, [ptr0_1]
        add ptr0_1, ptr0_1, #4
        trn1 v0.2s, v0.2s, v1.2s
        st1 {v0.8b}, [ptr_dst], #8
    
    loopmd2_kd2:
        cmp kd2_ori, 0
        beq loopmd2_kd1
    
        ld1 {v0.8b}, [ptr0_0]
        add ptr0_0, ptr0_0, #2
        ld1 {v1.8b}, [ptr0_1]
        add ptr0_1, ptr0_1, #2
        trn1 v0.4h, v0.4h, v1.4h
        st1 {v0.8b}, [ptr_dst]
        add ptr_dst, ptr_dst, #4
    
    loopmd2_kd1:
        cmp kd1_ori, 0
        beq loopmd2_continue
    
        ldrb temp1, [ptr0_0], #1
        strb temp1, [ptr_dst], #1
        ldrb temp2, [ptr0_1], #1
        strb temp2, [ptr_dst], #1

        // update data pointers
    loopmd2_continue:
        mov ptr0_0, ptr0_1
        add ptr0_1, ptr0_0, k
        b loopmd_ln

// proc the last line
loopmd1_start:
    sub ml, ml, #1
    cmp kd8_ori, 0
    beq loopmd1_kd1

    mov kd, kd8_ori
    loopmd1_kd8:
	    ld1 {v0.8b}, [ptr0_0], #8
	    st1 {v0.8b}, [ptr_dst], #8
	    subs kd, kd, #1
	    bne loopmd1_kd8

    // left data size <= 7, copy it.
    loopmd1_kd1:
    cmp kd8_l, 0
    beq end 
    copy_kd8_l:
        ldrb temp1, [ptr0_0], #1
        strb temp1, [ptr_dst], #1
        subs kd8_l, kd8_l, 1
        bne copy_kd8_l
   
end:
    sub sp, sp, #64
    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [sp], #64
    ret

#endif
