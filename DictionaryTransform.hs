module DictionaryTransform where

import qualified Data.Map.Strict as M
import Data.Maybe (fromMaybe)

-- ============================================================
-- Σ : Alfabeto (simplificado)
-- ============================================================

type Sigma = Char

-- ============================================================
-- D : Dicionário (palavras independentes de contexto)
-- ============================================================

dictionary :: [String]
dictionary =
  [ "mesa"
  , "janela"
  , "pedra"
  , "caminho"
  , "chuva"
  , "peixe"
  , "montanha"
  , "livro"
  , "fogo"
  , "sombra"
  , "areia"
  , "vento"
  , "casa"
  , "folha"
  , "rio"
  , "cadeira"
  , "sol"
  , "noite"
  , "fruta"
  , "espelho"
  , "porta"
  , "estrada"
  , "copo"
  , "árvore"
  , "relógio"
  ]

-- ============================================================
-- f : Σ → Σ   (substituição determinística)
-- exemplo: a→A, b→B, ..., z→Z
-- ============================================================

charMap :: M.Map Sigma Sigma
charMap = M.fromList
  [ ('a','A'), ('b','B'), ('c','C'), ('d','D'), ('e','E')
  , ('f','F'), ('g','G'), ('h','H'), ('i','I'), ('j','J')
  , ('k','K'), ('l','L'), ('m','M'), ('n','N'), ('o','O')
  , ('p','P'), ('q','Q'), ('r','R'), ('s','S'), ('t','T')
  , ('u','U'), ('v','V'), ('w','W'), ('x','X'), ('y','Y')
  , ('z','Z')
  , ('á','Á'), ('é','É'), ('í','Í'), ('ó','Ó'), ('ú','Ú')
  , ('â','Â'), ('ê','Ê'), ('ô','Ô'), ('ã','Ã'), ('õ','Õ')
  , ('ç','Ç')
  ]

f :: Sigma -> Sigma
f c = fromMaybe c (M.lookup c charMap)

-- ============================================================
-- F : Σ* → Σ*   (lifting de f para palavras)
-- ============================================================

transformWord :: String -> String
transformWord = map f

-- ============================================================
-- S = { x ∈ D | F(x) ∈ Σ* }  (sempre verdadeiro aqui)
-- ============================================================

transformedDictionary :: [String]
transformedDictionary = map transformWord dictionary

-- ============================================================
-- Execução de teste
-- ============================================================

main :: IO ()
main = mapM_ print transformedDictionary
