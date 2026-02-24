import torch
import argparse

# Lista realista de ~100+ mnemonics x86/amd64 (amostra — pode expandir com Intel SDM ou felixcloutier)
MNEMONICS = [
    "AAA", "AAD", "AAM", "AAS", "ADC", "ADD", "ADDPD", "ADDPS", "ADDSD", "ADDSS",
    "ADDSUBPD", "ADDSUBPS", "AND", "ANDNPD", "ANDNPS", "ANDPD", "ANDPS",
    "BLENDPD", "BLENDPS", "BLENDVPD", "BLENDVPS", "BLSIC", "BSF", "BSR", "BSWAP",
    "BT", "BTC", "BTR", "BTS", "CALL", "CBW", "CDQ", "CDQE", "CLC", "CLD",
    "CLI", "CLTS", "CMC", "CMOVA", "CMOVAE", "CMOVB", "CMOVBE", "CMOVC",
    "CMOVE", "CMOVG", "CMOVGE", "CMOVL", "CMOVLE", "CMOVNA", "CMOVNB",
    "CMOVNE", "CMOVNO", "CMOVNP", "CMOVNS", "CMOVO", "CMOVP", "CMOVPE",
    "CMOVPO", "CMOVS", "CMP", "CMPPD", "CMPPS", "CMPS", "CMPSB", "CMPSD",
    "CMPSW", "CMPXCHG", "COMISD", "COMISS", "CPUID", "CVTDQ2PD", "CVTDQ2PS",
    "CVTPD2DQ", "CVTPD2PI", "CVTPD2PS", "CVTPI2PD", "CVTPI2PS", "CVTPS2DQ",
    "CVTPS2PD", "CVTPS2PI", "CVTSD2SI", "CVTSD2SS", "CVTSI2SD", "CVTSI2SS",
    "CVTSS2SD", "CVTSS2SI", "CVTTPD2DQ", "CVTTPD2PI", "CVTTPS2DQ", "CVTTPS2PI",
    "CVTTSD2SI", "CVTTSS2SI", "CWD", "CWDE", "DAA", "DAS", "DEC", "DIV", "DIVPD",
    "DIVPS", "DIVSD", "DIVSS", "EMMS", "ENCLS", "ENCLU", "ENTER", "EXTRACTPS",
    "F2XM1", "FABS", "FADD", "FADDP", "FBLD", "FBSTP", "FCHS", "FCLEX",
    "FCMOVB", "FCMOVBE", "FCMOVE", "FCMOVNB", "FCMOVNBE", "FCMOVNE",
    "FCMOVNU", "FCMOVU", "FCOM", "FCOMI", "FCOMIP", "FCOMP", "FCOMPP",
    "FCOS", "FDECSTP", "FDIV", "FDIVP", "FDIVR", "FDIVRP", "FEMMS", "FFREE",
    "FIADD", "FICOM", "FICOMP", "FIDIV", "FIDIVR", "FILD", "FIMUL", "FINCSTP",
    "FNINIT", "FIST", "FISTP", "FISTTP", "FISUB", "FISUBR", "FLD", "FLD1",
    "FLDCW", "FLDENV", "FLDL2E", "FLDL2T", "FLDLG2", "FLDLN2", "FLDPI", "FLDZ",
    "FMUL", "FMULP", "FNCLEX", "FNOP", "FNSAVE", "FNSTCW", "FNSTENV", "FNSTSW",
    "FPATAN", "FPREM", "FPREM1", "FPTAN", "FRNDINT", "FRSTOR", "FSAVE",
    "FSCALE", "FSIN", "FSINCOS", "FSQRT", "FST", "FSTCW", "FSTENV", "FSTP",
    "FSTSW", "FSUB", "FSUBP", "FSUBR", "FSUBRP", "FTST", "FUCOM", "FUCOMI",
    "FUCOMIP", "FUCOMP", "FUCOMPP", "FWAIT", "FXAM", "FXCH", "FXRSTOR",
    "FXSAVE", "FXTRACT", "FYL2X", "FYL2XP1", "HLT", "IDIV", "IMUL", "IN",
    "INC", "INS", "INSB", "INSD", "INSW", "INT", "INT3", "INTO", "INVD",
    "INVLPG", "INVPCID", "IRET", "IRETD", "IRETQ", "JA", "JAE", "JB", "JBE",
    "JC", "JCXZ", "JE", "JECXZ", "JG", "JGE", "JL", "JLE", "JMP", "JNA",
    "JNAE", "JNB", "JNBE", "JNC", "JNE", "JNG", "JNGE", "JNL", "JNLE", "JNO",
    "JNP", "JNS", "JNZ", "JO", "JP", "JPE", "JPO", "JS", "JZ", "LAHF", "LAR",
    "LDDQU", "LDMXCSR", "LEA", "LEAVE", "LFENCE", "LGDT", "LIDT", "LLDT",
    "LMSW", "LODS", "LODSB", "LODSD", "LODSQ", "LODSW", "LOOP", "LOOPE",
    "LOOPNE", "LOOPNZ", "LOOPZ", "LSL", "LSS", "LTR", "MASKMOVDQU",
    "MASKMOVQ", "MAXPD", "MAXPS", "MAXSD", "MAXSS", "MFENCE", "MINPD",
    "MINPS", "MINSD", "MINSS", "MONITOR", "MOV", "MOVAPD", "MOVAPS", "MOVD",
    "MOVDDUP", "MOVDQ2Q", "MOVDQA", "MOVDQU", "MOVHLPS", "MOVHPD", "MOVHPS",
    "MOVLHPS", "MOVLPD", "MOVLPS", "MOVMSKPD", "MOVMSKPS", "MOVNTDQ",
    "MOVNTDQA", "MOVNTI", "MOVNTPD", "MOVNTPS", "MOVNTQ", "MOVQ", "MOVQ2DQ",
    "MOVS", "MOVSB", "MOVSD", "MOVSHDUP", "MOVSLDUP", "MOVSS", "MOVSW",
    "MOVSX", "MOVSXD", "MOVUPD", "MOVUPS", "MOVZX", "MPSADBW", "MUL", "MULPD",
    "MULPS", "MULSD", "MULSS", "MWAIT", "NEG", "NOP", "NOT", "OR", "ORPD",
    "ORPS", "OUT", "OUTS", "OUTSB", "OUTSD", "OUTSW", "PABSB", "PABSD",
    "PABSW", "PACKSSDW", "PACKSSWB", "PACKUSDW", "PACKUSWB", "PADDB", "PADDD",
    "PADDQ", "PADDSB", "PADDSW", "PADDUSB", "PADDUSW", "PADDW", "PALIGNR",
    "PAND", "PANDN", "PAUSE", "PAVGB", "PAVGUSB", "PAVGW", "PBLENDVB",
    "PBLENDW", "PCLMULQDQ", "PCMPEQB", "PCMPEQD", "PCMPEQW", "PCMPESTRI",
    "PCMPESTRM", "PCMPGTB", "PCMPGTD", "PCMPGTW", "PCMPISTRI", "PCMPISTRM",
    "PEXTRB", "PEXTRD", "PEXTRQ", "PEXTRW", "PHADDD", "PHADDSW", "PHADDW",
    "PHMINPOSUW", "PHSUBD", "PHSUBSW", "PHSUBW", "PINSRB", "PINSRD",
    "PINSRQ", "PINSRW", "PMADDUBSW", "PMADDWD", "PMAXSB", "PMAXSD", "PMAXSW",
    "PMAXUB", "PMAXUD", "PMAXUW", "PMINSB", "PMINSD", "PMINSW", "PMINUB",
    "PMINUD", "PMINUW", "PMOVMSKB", "PMOVSXBD", "PMOVSXBQ", "PMOVSXBW",
    "PMOVSXDQ", "PMOVSXWD", "PMOVSXWQ", "PMOVZXBD", "PMOVZXBQ", "PMOVZXBW",
    "PMOVZXDQ", "PMOVZXWD", "PMOVZXWQ", "PMULDQ", "PMULHRSW", "PMULHUW",
    "PMULHW", "PMULLD", "PMULLW", "PMULUDQ", "POP", "POPA", "POPAD", "POPCNT",
    "POPF", "POPFD", "POPFQ", "POR", "PREFETCHNTA", "PREFETCHT0",
    "PREFETCHT1", "PREFETCHT2", "PREFETCHW", "PSADBW", "PSHUFB", "PSHUFD",
    "PSHUFHW", "PSHUFLW", "PSHUFW", "PSIGNB", "PSIGND", "PSIGNW", "PSLLD",
    "PSLLDQ", "PSLLQ", "PSLLW", "PSRAD", "PSRAW", "PSRLD", "PSRLDQ", "PSRLQ",
    "PSRLW", "PSUBB", "PSUBD", "PSUBQ", "PSUBSB", "PSUBSW", "PSUBUSB",
    "PSUBUSW", "PSUBW", "PTEST", "PUNPCKHBW", "PUNPCKHDQ", "PUNPCKHQDQ",
    "PUNPCKHWD", "PUNPCKLBW", "PUNPCKLDQ", "PUNPCKLQDQ", "PUNPCKLWD", "PUSH",
    "PUSHA", "PUSHAD", "PUSHF", "PUSHFD", "PUSHFQ", "PXOR", "RCL", "RCPPS",
    "RCPSS", "RCR", "RDMSR", "RDPMC", "RDTSC", "RDTSCP", "REP", "REPE", "REPNE",
    "REPNZ", "RET", "ROL", "ROR", "RORX", "ROUNDPD", "ROUNDPS", "ROUNDSD",
    "ROUNDSS", "RSM", "RSQRTPS", "RSQRTSS", "SAHF", "SAL", "SAR", "SARX",
    "SBB", "SCAS", "SCASB", "SCASD", "SCASW", "SETA", "SETAE", "SETB", "SETBE",
    "SETC", "SETE", "SETG", "SETGE", "SETL", "SETLE", "SETNA", "SETNAE",
    "SETNB", "SETNBE", "SETNC", "SETNE", "SETNG", "SETNGE", "SETNL", "SETNLE",
    "SETNO", "SETNP", "SETNS", "SETNZ", "SETO", "SETP", "SETPE", "SETPO",
    "SETS", "SETZ", "SFENCE", "SGDT", "SHL", "SHLD", "SHR", "SHRD", "SHRX",
    "SHUFPD", "SHUFPS", "SIDT", "SKINIT", "SLDT", "SMSW", "SQRTPD", "SQRTPS",
    "SQRTSD", "SQRTSS", "STC", "STD", "STI", "STMXCSR", "STOS", "STOSB",
    "STOSD", "STOSQ", "STOSW", "STR", "SUB", "SUBPD", "SUBPS", "SUBSD", "SUBSS",
    "SWAPGS", "SYSCALL", "SYSENTER", "SYSEXIT", "SYSRET", "TEST", "UCOMISD",
    "UCOMISS", "UD2", "UNPCKHPD", "UNPCKHPS", "UNPCKLPD", "UNPCKLPS", "VERR",
    "VERW", "VMCALL", "VMCLEAR", "VMLAUNCH", "VMPTRLD", "VMPTRST", "VMREAD",
    "VMRESUME", "VMWRITE", "VMXOFF", "VMXON", "WBINVD", "WRMSR", "XABORT",
    "XACQUIRE", "XADD", "XBEGIN", "XCHG", "XEND", "XGETBV", "XLAT", "XLATB",
    "XOR", "XORPD", "XORPS", "XRSTOR", "XSAVE", "XSETBV", "XTEST"
]

class MnemonicTracer:
    def __init__(self, mnemonics, device="cpu"):
        self.device = torch.device(device)
        self.mnemonics = mnemonics

    def build_state(self, N: int) -> torch.Tensor:
        nm1 = N - 1
        return torch.tensor([nm1**2, float(nm1), float(N)], device=self.device)

    def reflect(self, state: torch.Tensor, idx: int = 2) -> torch.Tensor:
        state = state.clone()
        state[idx] = 2 * state[idx] - 1
        return state

    def expand(self, state: torch.Tensor, sequence=None) -> torch.Tensor:
        if sequence is None:
            sequence = [0, 1, 2, 3]
        seq = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        expanded = [(state[i] + seq)**2 for i in range(len(state))]
        return torch.cat(expanded)

    def project(self, state: torch.Tensor) -> float:
        return state.sum().item()

    def run_tracert(self, max_items: int = 40):
        print(f"{'#':3} | {'Mnemonic':12} | {'N':4} | {'Proj':12} | {'State (expanded preview)'}")
        print("-" * 100)
        for idx, mnem in enumerate(self.mnemonics[:max_items], start=1):
            state = self.build_state(idx)
            reflected = self.reflect(state)
            expanded = self.expand(reflected)
            proj = self.project(expanded)
            vec_preview = ", ".join(f"{int(x.item())}" for x in expanded[:8])
            if len(expanded) > 8:
                vec_preview += ", …"
            print(f"{idx:3d} | {mnem:12} | {idx:4d} | {proj:12.0f} | {vec_preview}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mnemonic Tracer Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the tracer simulation")
    parser.add_argument("--max_items", type=int, default=50, help="Maximum number of items to trace")
    args = parser.parse_args()

    if args.simulate or not any(vars(args).values()):
        tracer = MnemonicTracer(MNEMONICS)
        tracer.run_tracert(max_items=args.max_items)
