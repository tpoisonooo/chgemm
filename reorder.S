#ifdef __aarch64__

#define  ptr_src  x0
#define  ptr_dst  x13
#define  m  x2
#define  k  x3
#define  ptr0_0  x4
#define  ptr0_1  x5
#define  ptr0_2  x6
#define  ptr0_3  x7
#define  md4  x8
#define  kd8  x9
#define  kd8_ori  x10
#define  ml4  x11
#define  ml4_l  x12
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
subs ml4, m, md4
lsr md4, m, 2
lsr kd8, k, 3
mov kd8_ori, kd8
mov ptr_dst, x1
mov ptr0_0, ptr_src
add ptr0_1, ptr0_0, k
add ptr0_2, ptr0_1, k
add ptr0_3, ptr0_2, k
subs md4, md4, #0
beq loopmd4_ln
loopmd4:
	mov kd8, kd8_ori
loopkd8:
    ld1 {v0.8b}, [ptr0_0], #8
    ld1 {v1.8b}, [ptr0_1], #8
    ld1 {v2.8b}, [ptr0_2], #8
    ld1 {v3.8b}, [ptr0_3], #8
    st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [ptr_dst], #32
    subs kd8, kd8, #1
    bne loopkd8

	mov ptr0_0, ptr0_3
	add ptr0_1, ptr0_0, k
	add ptr0_2, ptr0_1, k
	add ptr0_3, ptr0_2, k
	subs md4, md4, #1
	bne loopmd4

loopmd4_ln:
	subs ml4_l, ml4, #0
	beq end
	subs ml4_l, ml4, #3
	mov kd8, kd8_ori
	beq loopmd4_l3
	subs ml4_l, ml4, #2
	mov kd8, kd8_ori
	beq loopmd4_l2
	subs ml4_l, ml4, #1
	mov kd8, kd8_ori
	beq loopmd4_l1

loopmd4_l3:
    ld1 {v0.8b}, [ptr0_0], #8
    ld1 {v1.8b}, [ptr0_1], #8
    ld1 {v2.8b}, [ptr0_2], #8
    st1 {v0.8b, v1.8b, v2.8b}, [ptr_dst], #24
    subs kd8, kd8, #1
    bne loopmd4_l3
    b end

loopmd4_l2:
	ld1 {v0.8b}, [ptr0_0], #8
	ld1 {v1.8b}, [ptr0_1], #8
	st1 {v0.8b, v1.8b}, [ptr_dst], #16
	subs kd8, kd8, #1
	bne loopmd4_l2
    b end

loopmd4_l1:
	ld1 {v0.8b}, [ptr0_0], #8
	st1 {v0.8b}, [ptr_dst], #8
	subs kd8, kd8, #1
	bne loopmd4_l1
    b end

end:
sub sp, sp, #64
ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [sp], #64
ret

#endif
