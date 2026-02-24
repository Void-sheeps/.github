import torch
import torch.nn as nn
import argparse
import json

# Lista realista de ~100+ mnemonics x86/amd64
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

def build_n2_matrix(N: int, device="cpu") -> torch.Tensor:
    """
    Expande N^2 em vetores R^3 como definido pelo usuário.
    """
    nm1_sq = float((N - 1)**2)
    row1 = [nm1_sq, float(N - 1), float(N)]
    row2 = [nm1_sq, float(2 * N), -1.0]
    row3 = [nm1_sq, 0.0, float(2 * (N - 0.5))]
    row4 = [nm1_sq, float(2 * (N - 0.5)), 0.0]

    return torch.tensor([row1, row2, row3, row4], dtype=torch.float32, device=device)

class N2ReferenceEmbedding(nn.Module):
    """
    Embedding determinístico baseado na expansão N².
    Projeta o vetor 12D para d_model.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Projeção linear para o espaço d_model
        self.projection = nn.Linear(12, d_model)

    def forward(self, N_tokens: torch.Tensor):
        # N_tokens: tensor de índices [batch_size, seq_len]
        batch_size, seq_len = N_tokens.shape
        embeddings = []

        for b in range(batch_size):
            seq_embeddings = []
            for s in range(seq_len):
                N = N_tokens[b, s].item() + 1 # +1 pois N começa em 1 na lógica do usuário
                matrix = build_n2_matrix(N, device=N_tokens.device)
                flattened = matrix.flatten()
                seq_embeddings.append(flattened)
            embeddings.append(torch.stack(seq_embeddings))

        x = torch.stack(embeddings) # [batch_size, seq_len, 12]
        return self.projection(x)

class MnemonicTransformer(nn.Module):
    """
    Sistema de processamento de mnemônicos x86.
    Integra N2ReferenceEmbedding com camadas de atenção.
    """
    def __init__(self, d_model: int, nhead: int, num_layers: int):
        super().__init__()
        self.embedding = N2ReferenceEmbedding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, N_tokens: torch.Tensor):
        x = self.embedding(N_tokens)
        output = self.transformer(x)
        return self.fc_out(output)

class MnemonicTracer:
    def __init__(self, mnemonics, device="cpu"):
        self.device = torch.device(device)
        self.mnemonics = mnemonics

    def run_tracert(self, max_items: int = 40, export_json: bool = False):
        print(f"{'#':3} | {'Mnemonic':12} | {'N':4} | {'Proj':12} | {'Flattened Preview'}")
        print("-" * 100)

        token_table = {}
        for idx, mnem in enumerate(self.mnemonics[:max_items], start=1):
            matrix = build_n2_matrix(idx, device=self.device)
            flattened = matrix.flatten()
            proj = matrix.sum().item()

            vec_preview = ", ".join(f"{x.item():.1f}" for x in flattened[:6])
            print(f"{idx:3d} | {mnem:12} | {idx:4d} | {proj:12.1f} | {vec_preview}, …")

            if export_json:
                token_table[f"N{idx}"] = {
                    "mnemonic": mnem,
                    "matrix": matrix.tolist(),
                    "vector": flattened.tolist(),
                    "projection": proj
                }

        if export_json:
            with open("mnemonic_data.json", "w") as f:
                json.dump(token_table, f, indent=2)
            print(f"\nResults exported to mnemonic_data.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mnemonic Tracer and Transformer Simulation")
    parser.add_argument("--simulate", action="store_true", help="Run the tracer simulation")
    parser.add_argument("--json", action="store_true", help="Export results to mnemonic_data.json")
    parser.add_argument("--transformer", action="store_true", help="Run MnemonicTransformer forward pass")
    parser.add_argument("--max_items", type=int, default=50, help="Maximum number of items to trace")
    args = parser.parse_args()

    if args.transformer:
        print("\n--- MnemonicTransformer Simulation ---")
        d_model = 512
        nhead = 8
        num_layers = 4
        model = MnemonicTransformer(d_model, nhead, num_layers)

        # Exemplo: batch de 2 sequências de 5 mnemônicos cada
        # Índices aleatórios entre 0 e len(MNEMONICS)-1
        sample_tokens = torch.randint(0, len(MNEMONICS), (2, 5))

        output = model(sample_tokens)
        print(f"Input Tokens (Indices):\n{sample_tokens}")
        print(f"Output Shape: {output.shape} (batch, seq, d_model)")
        print(f"Sample Output (first sequence, first token, first 8 components):\n{output[0, 0, :8]}")

    if args.simulate or args.json or (not args.transformer and not any(vars(args).values())):
        tracer = MnemonicTracer(MNEMONICS)
        tracer.run_tracert(max_items=args.max_items, export_json=args.json or args.simulate)
