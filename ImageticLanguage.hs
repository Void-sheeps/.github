module ImageticLanguage where

import qualified Data.Map.Strict as M
import Data.List (foldl')

-- ============================================================
-- 1. Palavra (símbolo puro, sem peso)
-- ============================================================

type WordSymbol = String

-- ============================================================
-- 2. Imagem = vetor abstrato (posição, não significado)
-- ============================================================

type Image = [Double]

-- ============================================================
-- 3. Associação imagética (peso está na relação)
-- ============================================================

type Association = Double

-- ============================================================
-- 4. Dicionário: palavra → imagem
-- (imagem não carrega semântica, só coordenada)
-- ============================================================

type Dictionary = M.Map WordSymbol Image

dictionary :: Dictionary
dictionary = M.fromList
  [ ("mesa",     [1.0, 0.2, 0.0])
  , ("cadeira",  [0.9, 0.3, 0.1])
  , ("janela",   [0.2, 1.0, 0.4])
  , ("porta",    [0.3, 0.9, 0.5])
  , ("pedra",    [0.8, 0.1, 0.9])
  ]

-- ============================================================
-- 5. Relações imagéticas (onde o peso realmente existe)
-- ============================================================

type Relations = M.Map (WordSymbol, WordSymbol) Association

relations :: Relations
relations = M.fromList
  [ (("mesa","cadeira"), 0.8)
  , (("mesa","janela"),  0.2)
  , (("porta","janela"), 0.7)
  , (("pedra","porta"),  0.1)
  ]

-- ============================================================
-- 6. Operações vetoriais (imagem pura)
-- ============================================================

scale :: Double -> Image -> Image
scale a = map (* a)

add :: Image -> Image -> Image
add = zipWith (+)

zeroImage :: Int -> Image
zeroImage n = replicate n 0.0

-- ============================================================
-- 7. Contexto = imagem resultante da associação
-- ============================================================

contextImage
  :: Dictionary
  -> Relations
  -> WordSymbol
  -> Image
contextImage dict rels w =
  foldl' add baseImage associatedImages
  where
    baseImage =
      M.findWithDefault (zeroImage dim) w dict

    dim = length baseImage

    associatedImages =
      [ scale weight img
      | ((w1, w2), weight) <- M.toList rels
      , w1 == w
      , Just img <- [M.lookup w2 dict]
      ]

-- ============================================================
-- 8. Exemplo de uso
-- ============================================================

example :: Image
example = contextImage dictionary relations "mesa"
