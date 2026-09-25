00000fb8: lea ecx, [rbx + r8*2 + 0x1800]
00000fc0: mov rdx, qword ptr [rdi + 0x38]
00000fc4: vmovdqu xmm5, xmmword ptr [rdx + rcx + 0x60]
00000fca: lea edx, [r10 + r8]
00000fce: mov rsi, qword ptr [rdi + 0x38]
00000fd2: vpmovsxbw xmm4, qword ptr [rsi + rdx + 0x30]
00000fd9: vpmaddwd xmm0, xmm5, xmm4
00000fdd: movdqu xmmword ptr [rsp + 0x1a40], xmm4
00000fe6: mov rsi, qword ptr [rdi + 0x38]
00000fea: vmovdqu xmm6, xmmword ptr [rsi + rcx + 0x50]
00000ff0: mov rsi, qword ptr [rdi + 0x38]
00000ff4: vpmovsxbw xmm1, qword ptr [rsi + rdx + 0x28]
00000ffb: vpmaddwd xmm3, xmm6, xmm1
00000fff: movdqu xmmword ptr [rsp + 0x1a20], xmm1
00001008: vpaddd xmm0, xmm0, xmm3
0000100c: mov rsi, qword ptr [rdi + 0x38]
00001010: vmovdqu xmm9, xmmword ptr [rsi + rcx + 0x70]
00001016: mov rsi, qword ptr [rdi + 0x38]
0000101a: vpmovsxbw xmm7, qword ptr [rsi + rdx + 0x38]
00001021: vpmaddwd xmm1, xmm9, xmm7
00001025: movdqu xmmword ptr [rsp + 0x1a00], xmm7
0000102e: vpaddd xmm0, xmm0, xmm1
00001032: mov rsi, qword ptr [rdi + 0x38]
00001036: vmovdqu xmm8, xmmword ptr [rsi + rcx]
0000103b: movdqu xmmword ptr [rsp + 0x19f0], xmm8
00001045: mov rsi, qword ptr [rdi + 0x38]
00001049: vpmovsxbw xmm11, qword ptr [rsi + rdx]
0000104f: vpmaddwd xmm3, xmm8, xmm11
00001054: movdqu xmm7, xmmword ptr [rsp + 0xb0]
0000105d: movdqu xmmword ptr [rsp + 0x19e0], xmm11
00001067: vpaddd xmm4, xmm3, xmm7
0000106b: mov rsi, qword ptr [rdi + 0x38]
0000106f: vmovdqu xmm3, xmmword ptr [rsi + rcx + 0x10]
00001075: mov rsi, qword ptr [rdi + 0x38]
00001079: vpmovsxbw xmm12, qword ptr [rsi + rdx + 8]
00001080: vpmaddwd xmm7, xmm3, xmm12
00001085: movdqu xmmword ptr [rsp + 0x19c0], xmm12
0000108f: vpaddd xmm4, xmm4, xmm7
00001093: mov rsi, qword ptr [rdi + 0x38]
00001097: vmovdqu xmm8, xmmword ptr [rsi + rcx + 0x20]
0000109d: mov rsi, qword ptr [rdi + 0x38]
000010a1: vpmovsxbw xmm13, qword ptr [rsi + rdx + 0x10]
000010a8: vpmaddwd xmm7, xmm8, xmm13
000010ad: movdqu xmmword ptr [rsp + 0x19a0], xmm13
000010b7: vpaddd xmm7, xmm4, xmm7
000010bb: mov rsi, qword ptr [rdi + 0x38]
000010bf: vmovdqu xmm4, xmmword ptr [rsi + rcx + 0x30]
000010c5: mov rsi, qword ptr [rdi + 0x38]
000010c9: vpmovsxbw xmm11, qword ptr [rsi + rdx + 0x18]
000010d0: vpmaddwd xmm12, xmm4, xmm11
000010d5: movdqu xmmword ptr [rsp + 0x1980], xmm11
000010df: vpaddd xmm7, xmm7, xmm12
000010e4: vpaddd xmm0, xmm0, xmm7
000010e8: mov rsi, qword ptr [rdi + 0x38]
000010ec: vmovdqu xmm13, xmmword ptr [rsi + rcx + 0x40]
000010f2: mov rcx, qword ptr [rdi + 0x38]
000010f6: vpmovsxbw xmm7, qword ptr [rcx + rdx + 0x20]
000010fd: vpmaddwd xmm11, xmm13, xmm7
00001101: movdqu xmmword ptr [rsp + 0x1960], xmm7
0000110a: vpaddd xmm7, xmm0, xmm11
0000110f: movdqu xmmword ptr [rsp + 0xb0], xmm7
00001118: lea ecx, [rbx + r8*2 + 0x4800]
00001120: mov rdx, qword ptr [rdi + 0x38]
00001124: vmovdqu xmm7, xmmword ptr [rdx + rcx + 0x70]
0000112a: lea edx, [r10 + r8 + 0x900]
00001132: mov rsi, qword ptr [rdi + 0x38]
00001136: vpmovsxbw xmm11, qword ptr [rsi + rdx + 0x38]
0000113d: vpmaddwd xmm0, xmm7, xmm11
00001142: movdqu xmmword ptr [rsp + 0x1940], xmm11
0000114c: movdqu xmmword ptr [rsp + 0x1950], xmm7
00001155: mov rsi, qword ptr [rdi + 0x38]
00001159: vmovdqu xmm7, xmmword ptr [rsi + rcx + 0x60]
0000115f: mov rsi, qword ptr [rdi + 0x38]
00001163: vpmovsxbw xmm11, qword ptr [rsi + rdx + 0x30]
0000116a: vpmaddwd xmm12, xmm7, xmm11
0000116f: movdqu xmmword ptr [rsp + 0x1920], xmm11
00001179: movdqu xmmword ptr [rsp + 0x1930], xmm7
00001182: vpaddd xmm0, xmm0, xmm12
00001187: mov rsi, qword ptr [rdi + 0x38]
0000118b: vmovdqu xmm15, xmmword ptr [rsi + rcx]
