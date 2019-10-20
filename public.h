#if __aarch64__

.macro ASM_DECLARE function_name
#ifdef __APPLE__
.globl _\function_name
_\function_name:
#else
.global \function_name
#ifdef __ELF__
.hidden \function_name
.type \function_name, %function
#endif
\function_name:
#endif
.endm

.macro SAVE_REGS
    sub sp, sp, #304
    stp d8, d9, [sp], #16
    stp d10, d11, [sp], #16
    stp d12, d13, [sp], #16
    stp d14, d15, [sp], #16
    stp d16, d17, [sp], #16

    stp x18, x19, [sp], #16
    stp x20, x21, [sp], #16
    stp x22, x23, [sp], #16
    stp x24, x25, [sp], #16
    stp x26, x27, [sp], #16
    str x28, [sp], #16

    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
.endm

.macro RESTORE_REGS
    sub sp, sp, #304
    ldp d8, d9, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d14, d15, [sp], #16
    ldp d16, d17, [sp], #16
    ldp x18, x19, [sp], #16
    ldp x20, x21, [sp], #16
    ldp x22, x23, [sp], #16
    ldp x24, x25, [sp], #16
    ldp x26, x27, [sp], #16
    ldr x28, [sp], #16

    ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
    ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
.endm

#endif
