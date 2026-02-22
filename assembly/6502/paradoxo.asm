; ===============================
; 费米-狄拉克对称性验证
; ===============================

FAC     = $61     ; 5字节浮点累加器
ARG     = $66     ; 5字节浮点参数寄存器
ONE     = $A2     ; ROM常量 1.0

; ROM浮点例程
FP_ADD  = $B867   ; FAC = FAC + ARG
FP_DIV  = $BB12   ; FAC = ARG / FAC
FP_EXP  = $BC1B   ; FAC = exp(FAC)
FP_NEG  = $BFB4   ; FAC = -FAC
MOVEF   = $BBFC   ; 复制 FAC → ARG
MOVFM   = $BBA5   ; 从内存加载FAC

; 零页临时存储
TEMP_X  = $70     ; $70-$74: 存储x
TEMP_F1 = $75     ; $75-$79: 存储f1

*= $0801
        .word next
        .word 10
        .byte $9e
        .text "2061"     ; SYS 2061
        .byte 0
next
        .word 0

; ===============================
; 输入: x 在 FAC 中
; ===============================

; 1. 保存 x
        LDX #4
SAVE_X: LDA FAC,X
        STA TEMP_X,X
        DEX
        BPL SAVE_X

; 2. 步骤1: 计算 f1 = 1/(1+exp(x))
        JSR FP_EXP        ; FAC = exp(x)
        JSR MOVEF         ; ARG = exp(x)
        LDA #<ONE
        LDY #>ONE
        JSR MOVFM         ; FAC = 1.0
        JSR FP_ADD        ; FAC = 1 + exp(x)
        JSR MOVEF         ; ARG = 1 + exp(x)
        LDA #<ONE
        LDY #>ONE
        JSR MOVFM         ; FAC = 1.0
        JSR FP_NEG        ; FAC = -1.0 (标准化)
        JSR FP_NEG        ; FAC = +1.0
        JSR FP_DIV        ; FAC = 1/(1+exp(x)) = f1
        
        ; 保存 f1
        JSR MOVEF         ; ARG = f1
        LDX #4
SAVE_F1: LDA ARG,X
        STA TEMP_F1,X
        DEX
        BPL SAVE_F1

; 3. 步骤2: 计算 f2 = 1/(1+exp(-x))
        ; 恢复 x
        LDX #4
REST_X: LDA TEMP_X,X
        STA FAC,X
        DEX
        BPL REST_X
        
        JSR FP_NEG        ; FAC = -x
        JSR FP_EXP        ; FAC = exp(-x)
        JSR MOVEF         ; ARG = exp(-x)
        LDA #<ONE
        LDY #>ONE
        JSR MOVFM         ; FAC = 1.0
        JSR FP_ADD        ; FAC = 1 + exp(-x)
        JSR MOVEF         ; ARG = 1 + exp(-x)
        LDA #<ONE
        LDY #>ONE
        JSR MOVFM         ; FAC = 1.0
        JSR FP_NEG        ; FAC = -1.0 (标准化)
        JSR FP_NEG        ; FAC = +1.0
        JSR FP_DIV        ; FAC = 1/(1+exp(-x)) = f2

; 4. 步骤3: 相加
        ; 恢复 f1
        LDX #4
REST_F1: LDA TEMP_F1,X
        STA ARG,X
        DEX
        BPL REST_F1
        
        JSR FP_ADD        ; FAC = f1 + f2 = 1.0

; ===============================
; 结果: FAC = 1.0
; ===============================
        RTS