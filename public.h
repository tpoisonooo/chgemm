.macro ASM_DECLARE function_name
.global \function_name
#ifdef __ELF__
.hidden \function_name
.type \function_name, %function
#endif
\function_name:
.endm
