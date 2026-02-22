; mapa_ontologico_x64_alinhado.asm
; NASM (x86-64, System V ABI) implementation with correct stack alignment
; Build:
;   nasm -felf64 mapa_ontologico_x64_alinhado.asm -o mapa.o
;   gcc -no-pie mapa.o -o mapa
; Run:
;   ./mapa

global main
extern printf
extern exit

section .data
    ; Mensagens
    msg_silent      db "SILENTIUM: Potencialidade pura detectada.", 10, 0
    msg_estab       db "ESTABILIZADO (Natural)", 10, 0
    msg_trans       db "TRANSCENDIDO (Construido)", 10, 0
    msg_overflow    db "ERRO: Limite Estrutural Atingido", 10, 0

    fmt_state       db "P = %d", 10, "E_Final = %d", 10, "Status_Hex = 0x%02X", 10, "Categoria = %s", 10, 0
    cat_natural     db "Natural", 0
    cat_trans       db "Transcendental", 0

section .bss
    ; Estado (dword para simplicidade)
    p           resd 1
    eCorrente   resd 1
    status      resd 1
    eFinal      resd 1

section .text

; -----------------------------------------------------------------------------
; safe_printf(format, arg1, arg2, arg3, arg4)
;   Garante alinhamento local antes de chamar printf.
;   Não usa push rbp/pop rbp para evitar frames duplicados.
;   SysV: rdi=format, rsi=arg1, rdx=arg2, rcx=arg3, r8=arg4
; -----------------------------------------------------------------------------
safe_printf:
    sub rsp, 8          ; alinha RSP a 16 bytes para a chamada
    xor eax, eax        ; clear xmm regs count per ABI (variadic)
    call printf
    add rsp, 8
    ret

; -----------------------------------------------------------------------------
; processar(p, eInicial)
;   rdi = p, rsi = eInicial
; -----------------------------------------------------------------------------
processar:
    push rbp
    mov rbp, rsp

    ; salvar p,eCorrente em memória
    mov dword [p], edi
    mov dword [eCorrente], esi

    ; if p == 0 -> SILENTIUM
    cmp dword [p], 0
    je .silentium

    ; decide fluxo: se eCorrente < Q_LIMIAR (15) -> corrigir else estabilizar
    mov eax, dword [eCorrente]
    cmp eax, 0x0F
    jl .corrigir
    jge .estabilizar

.silentium:
    lea rdi, [rel msg_silent]
    call safe_printf
    mov eax, 0
    pop rbp
    ret

; -----------------------------------------------------------------------------
; ESTABILIZAR
; -----------------------------------------------------------------------------
.estabilizar:
    ; grau = (eCorrente - Q) & 0x7F
    mov eax, dword [eCorrente]
    sub eax, 0x0F
    and eax, 0x7F
    mov dword [status], eax
    mov eax, dword [eCorrente]
    mov dword [eFinal], eax

    ; print message
    lea rdi, [rel msg_estab]
    call safe_printf

    jmp .exibir_estado_and_ret

; -----------------------------------------------------------------------------
; CORRIGIR  (ordem: verificar E_MAX primeiro, depois Q_LIMIAR)
; -----------------------------------------------------------------------------
.corrigir:
.loop_start:
    mov eax, dword [eCorrente]

    ; Verifica limite estrutural absoluto primeiro (E_MAX = 0x7F)
    cmp eax, 0x7F
    jae .overflow

    ; Se já atingiu limiar Q, sair
    cmp eax, 0x0F
    jge .after_loop

    ; Incrementa e persiste
    inc dword [eCorrente]
    jmp .loop_start

.after_loop:
    ; TRANSCENDER: status = ((eCorrente - Q) & 0x7F) | 0x80
    mov eax, dword [eCorrente]
    sub eax, 0x0F
    and eax, 0x7F
    or eax, 0x80
    mov dword [status], eax
    mov eax, dword [eCorrente]
    mov dword [eFinal], eax

    ; print message
    lea rdi, [rel msg_trans]
    call safe_printf

    jmp .exibir_estado_and_ret

; -----------------------------------------------------------------------------
; OVERFLOW
; -----------------------------------------------------------------------------
.overflow:
    ; status = 0x80 (categoria distinta, grau 0)
    mov dword [status], 0x80
    mov eax, dword [eCorrente]
    mov dword [eFinal], eax

    ; print message
    lea rdi, [rel msg_overflow]
    call safe_printf

    jmp .exibir_estado_and_ret

; -----------------------------------------------------------------------------
; Exibir estado e retornar
;   prepara argumentos para printf(fmt_state, P, E_Final, Status, Categoria)
; -----------------------------------------------------------------------------
.exibir_estado_and_ret:
    ; rdi = fmt_state
    lea rdi, [rel fmt_state]

    ; rsi = P
    mov eax, dword [p]
    mov esi, eax

    ; rdx = E_Final
    mov eax, dword [eFinal]
    mov edx, eax

    ; rcx = Status (mascarado para byte)
    mov eax, dword [status]
    and eax, 0xFF
    mov ecx, eax

    ; r8 = Categoria string
    mov eax, dword [status]
    test eax, 0x80
    jz .is_natural
    lea r8, [rel cat_trans]
    jmp .call_safe_printf

.is_natural:
    lea r8, [rel cat_natural]

.call_safe_printf:
    call safe_printf

    mov eax, 0
    pop rbp
    ret

; -----------------------------------------------------------------------------
; main: exemplo de uso com alinhamento garantido
; -----------------------------------------------------------------------------
main:
    push rbp
    mov rbp, rsp
    sub rsp, 8          ; garante RSP % 16 == 0 antes de chamadas C

    ; Exemplo: processar(1, 5)
    mov edi, 1      ; p
    mov esi, 5      ; eInicial
    call processar

    add rsp, 8       ; restaura stack antes de exit
    mov edi, 0
    call exit
