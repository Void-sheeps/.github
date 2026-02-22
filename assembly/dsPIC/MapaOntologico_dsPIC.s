; Mapa Ontológico para PIC24 / dsPIC (versão refinada)
; W0 = P
; W1 = E
; W2..W4 = registradores temporários
; _STATUS (word), _E_FINAL (word) em RAM

    .include "p24Fxxxx.inc"    ; substituir pelo header do dispositivo
    .section .data
_STATUS:    .space 2
_E_FINAL:   .space 2

    .section .text
    .global _start

_start:
    ; Inicialização de exemplo (usar mov.w se necessário)
    MOV #0x0001, W0       ; P = 1
    MOV #0x0005, W1       ; E = 5

    ; Teste SILENTIUM
    TST.W   W0
    BRZ     SILENTIUM

    ; Q_LIMIAR = 15
    MOV     #0x000F, W2
    ; Se E < Q_LIMIAR -> CORRIGIR
    CMP.W   W1, W2
    BRLT    CORRIGIR

; ---------------------------
; ESTABILIZAÇÃO NATURAL
; ---------------------------
ESTABILIZAR:
    ; W3 = E - Q
    SUB.W   W1, W2, W3        ; W3 = W1 - W2
    AND.W   #0x007F, W3       ; limpa bit7
    ; gravar em memória
    MOV.W   W3, _STATUS
    MOV.W   W1, _E_FINAL
    BRA     FINALIZAR

; ---------------------------
; CORREÇÃO E TRANSCENDÊNCIA
; ---------------------------
CORRIGIR:
    ; W2 = Q_LIMIAR (15) já carregado
    MOV     #0x007F, W4       ; E_MAX = 127

LOOP_CHECK:
    ; Verifica limite estrutural primeiro (E_MAX)
    CMP.W   W1, W4
    BRGE    OVERFLOW          ; se W1 >= 127 -> overflow

    ; Se já atingiu limiar Q, sair
    CMP.W   W1, W2
    BRGE    TRANSCENDER_CHECK ; se W1 >= Q -> já ok

    ; Incrementa E e persiste
    ADD.W   #1, W1
    BRA     LOOP_CHECK

TRANSCENDER_CHECK:
    ; Transcendência atingida (W1 >= Q)
    SUB.W   W1, W2, W3        ; W3 = E - Q
    IOR.W   #0x0080, W3       ; ativa bit7 (transcendental)
    MOV.W   W3, _STATUS
    MOV.W   W1, _E_FINAL
    BRA     FINALIZAR

OVERFLOW:
    MOV.W   #0x0080, W3       ; grau 0 + bit7
    MOV.W   W3, _STATUS
    MOV.W   W1, _E_FINAL
    BRA     FINALIZAR

SILENTIUM:
    IDLE                      ; modo baixo consumo
    BRA     SILENTIUM

FINALIZAR:
    RETURN
