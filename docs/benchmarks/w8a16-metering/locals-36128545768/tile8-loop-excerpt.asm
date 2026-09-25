00000940: lea esi, [rdx + 0x60]
00000943: mov r8, qword ptr [rdi + 0x38]
00000947: vmovdqu xmm14, xmmword ptr [r8 + rsi]
0000094d: lea esi, [rcx + rax]
00000950: lea r8d, [rcx + rax + 0xf30]
00000958: mov r11, qword ptr [rdi + 0x38]
0000095c: vpmovsxbw xmm10, qword ptr [r11 + r8]
00000962: vpmaddwd xmm0, xmm14, xmm10
00000967: movdqu xmmword ptr [rsp + 0xe20], xmm10
00000971: movdqu xmmword ptr [rsp + 0xe30], xmm14
0000097b: lea r8d, [rdx + 0x50]
0000097f: mov r11, qword ptr [rdi + 0x38]
00000983: vmovdqu xmm15, xmmword ptr [r11 + r8]
00000989: lea r8d, [rcx + rax + 0xf28]
00000991: mov r11, qword ptr [rdi + 0x38]
00000995: vpmovsxbw xmm10, qword ptr [r11 + r8]
0000099b: vpmaddwd xmm1, xmm15, xmm10
000009a0: movdqu xmmword ptr [rsp + 0xe00], xmm10
000009aa: movdqu xmmword ptr [rsp + 0xe10], xmm15
000009b4: vpaddd xmm0, xmm0, xmm1
000009b8: lea r8d, [rdx + 0x70]
000009bc: mov r11, qword ptr [rdi + 0x38]
000009c0: vmovdqu xmm3, xmmword ptr [r11 + r8]
000009c6: lea r8d, [rcx + rax + 0xf38]
000009ce: mov r11, qword ptr [rdi + 0x38]
000009d2: vpmovsxbw xmm10, qword ptr [r11 + r8]
000009d8: vpmaddwd xmm1, xmm3, xmm10
000009dd: movdqu xmmword ptr [rsp + 0xde0], xmm10
000009e7: movdqu xmmword ptr [rsp + 0xdf0], xmm3
000009f0: vpaddd xmm0, xmm0, xmm1
000009f4: mov r8d, edx
000009f7: mov r11, qword ptr [rdi + 0x38]
000009fb: vmovdqu xmm7, xmmword ptr [r11 + r8]
00000a01: lea r8d, [rcx + rax + 0xf00]
00000a09: mov r11, qword ptr [rdi + 0x38]
00000a0d: vpmovsxbw xmm10, qword ptr [r11 + r8]
00000a13: vpmaddwd xmm1, xmm7, xmm10
00000a18: movdqu xmm13, xmmword ptr [rsp + 0x120]
00000a22: movdqu xmmword ptr [rsp + 0xdc0], xmm10
00000a2c: movdqu xmmword ptr [rsp + 0xdd0], xmm7
00000a35: vpaddd xmm1, xmm1, xmm13
00000a3a: lea r8d, [rdx + 0x10]
00000a3e: mov r11, qword ptr [rdi + 0x38]
00000a42: vmovdqu xmm2, xmmword ptr [r11 + r8]
00000a48: lea r8d, [rcx + rax + 0xf08]
00000a50: mov r11, qword ptr [rdi + 0x38]
00000a54: vpmovsxbw xmm10, qword ptr [r11 + r8]
00000a5a: vpmaddwd xmm3, xmm2, xmm10
00000a5f: movdqu xmmword ptr [rsp + 0xda0], xmm10
00000a69: movdqu xmmword ptr [rsp + 0xdb0], xmm2
00000a72: vpaddd xmm1, xmm1, xmm3
00000a76: lea r8d, [rdx + 0x20]
00000a7a: mov r11, qword ptr [rdi + 0x38]
00000a7e: vmovdqu xmm2, xmmword ptr [r11 + r8]
00000a84: lea r8d, [rcx + rax + 0xf10]
00000a8c: mov r11, qword ptr [rdi + 0x38]
00000a90: vpmovsxbw xmm10, qword ptr [r11 + r8]
00000a96: vpmaddwd xmm3, xmm2, xmm10
00000a9b: movdqu xmmword ptr [rsp + 0xd80], xmm10
00000aa5: movdqu xmmword ptr [rsp + 0xd90], xmm2
00000aae: vpaddd xmm1, xmm1, xmm3
00000ab2: lea r8d, [rdx + 0x30]
00000ab6: mov r11, qword ptr [rdi + 0x38]
00000aba: vmovdqu xmm5, xmmword ptr [r11 + r8]
00000ac0: lea r8d, [rcx + rax + 0xf18]
00000ac8: mov r11, qword ptr [rdi + 0x38]
00000acc: vpmovsxbw xmm10, qword ptr [r11 + r8]
00000ad2: vpmaddwd xmm2, xmm5, xmm10
00000ad7: movdqu xmmword ptr [rsp + 0xd60], xmm10
00000ae1: movdqu xmmword ptr [rsp + 0xd70], xmm5
00000aea: vpaddd xmm1, xmm1, xmm2
00000aee: vpaddd xmm0, xmm0, xmm1
00000af2: lea r8d, [rdx + 0x40]
00000af6: mov r11, qword ptr [rdi + 0x38]
00000afa: vmovdqu xmm9, xmmword ptr [r11 + r8]
00000b00: lea r8d, [rcx + rax + 0xf20]
00000b08: mov r11, qword ptr [rdi + 0x38]
00000b0c: vpmovsxbw xmm10, qword ptr [r11 + r8]
00000b12: vpmaddwd xmm1, xmm9, xmm10
00000b17: movdqu xmmword ptr [rsp + 0xd40], xmm10
