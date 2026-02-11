# MapaOntologico.ps1
# Representação do Processador Ontológico em PowerShell

class MapaOntologico {
    # Constantes estruturais
    [int] $Q_LIMIAR = 0x0F   # 15
    [int] $E_MAX    = 0x7F   # 127

    # Memória (Estado Inicial)
    [int] $p = 0
    [int] $eCorrente = 0
    [int] $status = 0
    [int] $eFinal = 0

    MapaOntologico() {
        # Construtor vazio (valores já inicializados)
    }

    [string] Processar([int] $p, [int] $eInicial) {
        $this.p = $p
        $this.eCorrente = $eInicial

        # SILENTIUM: Se P = 0, entramos em estado de pausa (loop simulado)
        if ($this.p -eq 0) {
            Write-Host "SILENTIUM: Potencialidade pura detectada."
            return "Estado de Repouso"
        }

        # Decisão de Fluxo
        if ($this.eCorrente -lt $this.Q_LIMIAR) {
            return $this.Corrigir()
        } else {
            return $this.Estabilizar()
        }
    }

    [string] Estabilizar() {
        # grau = E - Q (Garante bit7 limpo como no Assembly)
        $grau = ($this.eCorrente - $this.Q_LIMIAR) -band 0x7F
        $this.status = $grau
        $this.eFinal = $this.eCorrente

        return "ESTABILIZADO (Natural)"
    }

    [string] Corrigir() {
        # Simulação da Região Crítica (SEI no Assembly)
        while ($this.eCorrente -lt $this.Q_LIMIAR) {
            if ($this.eCorrente -ge $this.E_MAX) {
                return $this.Overflow()
            }
            $this.eCorrente++
        }

        # TRANSCENDER: grau = E - Q | ativa bit 7 (0x80)
        $grau = ($this.eCorrente - $this.Q_LIMIAR) -band 0x7F
        $this.status = $grau -bor 0x80
        $this.eFinal = $this.eCorrente

        return "TRANSCENDIDO (Construído)"
    }

    [string] Overflow() {
        $this.status = 0x80   # Categoria distinta, grau 0
        $this.eFinal = $this.eCorrente
        return "ERRO: Limite Estrutural Atingido"
    }

    [hashtable] ExibirEstado() {
        $hex = ("0x{0:X2}" -f ($this.status -band 0xFF))
        $categoria = if (($this.status -band 0x80) -ne 0) { "Transcendental" } else { "Natural" }

        return @{
            P = $this.p
            E_Final = $this.eFinal
            Status_Hex = $hex
            Categoria = $categoria
        }
    }
}

# --- Exemplo de Uso ---
$sistema = [MapaOntologico]::new()

$result = $sistema.Processar(1, 5)   # Deve corrigir e transcender
Write-Host $result

# Exibe estado em tabela
$estado = $sistema.ExibirEstado()
$estado.GetEnumerator() | Sort-Object Name | Format-Table -AutoSize
