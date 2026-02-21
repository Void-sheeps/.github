module EmpireSilicium.VetorAbsoluto where

-- O Vetor Absoluto como um Tipo Algébrico de Dados
data Nota = C | Cs | D | Ds | E | F | Fs | G | Gs | A | As | B deriving (Enum, Bounded, Show)
data Oitava = O1 | O2 | O3 | O4 | O5 | O6 | O7 | O8 deriving (Enum, Bounded, Show)

-- O Acorde Inominável: O produto cartesiano de todas as possibilidades
-- Isso é o "Total Cromático" ou a "Divindade Semântica"
vetorAbsoluto :: [(Nota, Oitava)]
vetorAbsoluto = [ (n, o) | n <- [minBound..maxBound], o <- [minBound..maxBound] ]

-- Deus como uma Função de Ordem Superior
deus :: ([(Nota, Oitava)] -> a) -> a
deus interLegere = interLegere vetorAbsoluto
