* = $8000

; $0200 = P
; $0201 = E
; $0202 = STATUS (bit7 = categoria, bits0-6 = grau)
; $0203 = E final

START:
        LDA $0200
        BEQ SILENTIUM          ; Se P = 0, silêncio potencial

        LDX $0201              ; X = Evidência
        CPX #$0F               ; Regra Q (limiar)
        BCC CORRIGIR           ; Se E < Q, reestruturar

; ---------------------------
; ESTABILIZAÇÃO NATURAL
; ---------------------------
ESTABILIZAR:
        TXA                    ; A = E (X -> A)
        SEC
        SBC #$0F               ; grau = E - Q
        AND #%01111111         ; garante bit7 = 0 (natural)
        STA $0202              ; grava STATUS homogêneo
        STX $0203              ; persiste E final
        JMP FINALIZAR

; ---------------------------
; CORREÇÃO E TRANSCENDÊNCIA
; ---------------------------
CORRIGIR:
        SEI                    ; isolamento crítico

; Verifica se já está em/além do limite estrutural
        CPX #$7F
        BCS OVERFLOW           ; se X >= $7F, trata overflow/limite

LOOP_INC:
        INX                    ; X <- X + 1
        STX $0201              ; persiste E incrementada

; Se após incremento cruzou o limite estrutural, trate overflow
        CPX #$7F
        BCS OVERFLOW

        CPX #$0F
        BCC LOOP_INC

; Ao atingir limiar por construção → categoria distinta
        TXA                    ; A = E
        SEC
        SBC #$0F               ; grau = E - Q
        ORA #%10000000         ; ativa bit transcendental (bit7)
        STA $0202
        STX $0203
        CLI
        JMP FINALIZAR

OVERFLOW:
        LDA #%10000000         ; grau zero + bit distinto (ou outro código)
        STA $0202
        STX $0203
        CLI
        JMP FINALIZAR

SILENTIUM:
        JMP SILENTIUM

FINALIZAR:
        RTS
