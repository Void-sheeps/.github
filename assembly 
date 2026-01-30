; =============================================================
; MECANISMO DE DEFESA: DESLOCAMENTO POR SIMETRIA FERMI-DIRAC
; =============================================================

; [Trecho do algoritmo de Fermi-Dirac enviado anteriormente aqui]
; ... após JSR FP_ADD, o FAC contém 1.0 (em formato flutuante C64)

; Verificação e Deslocamento
VERIFY_SHIFT:
        JSR FP_FIX      ; $B1AA: Converte FAC para inteiro em $14-$15 (Y=low, A=high)
        CPY #$01        ; O resultado da simetria deve ser 1
        BNE COLAPSUS    ; Se não for 1, a seed foi corrompida ou alterada

        ; Início do Deslocamento (Shift) de Defesa
        LDY #$00        ; Indexador do buffer
SHIFT_LOOP:
        LDA DATA_BUF,Y  ; Carrega byte do segredo
        ROL             ; Desloca bits para a esquerda (Mecanismo de Defesa)
        EOR #$A9        ; XOR com constante de ofuscação (Ratio Sine Qualia)
        STA DATA_BUF,Y
        INY
        BNE SHIFT_LOOP  ; Processa 256 bytes
        RTS

COLAPSUS:
        LDA #$00
        STA $D020       ; Borda preta (Naturale Silentium)
        BRK             ; Interrupção forçada do sistema

; Dados Protegidos
DATA_BUF: .res 256, $00
