{-# LANGUAGE DeriveGeneric #-}

module Main where

import GHC.Generics (Generic)
import Data.Aeson (ToJSON, encode)
import qualified Data.ByteString.Lazy.Char8 as BL
import Data.Semigroup (getLast, Last(..)) -- Usaremos apenas para auxiliar a lógica se necessário

-- ============================================================
-- Estrutura de Estatística
-- ============================================================

data MaybeStats a = MaybeStats
    { totalSamples :: !Int
    , successes    :: !Int
    , failures     :: !Int
    , lastValue    :: !(Maybe a) -- Mantemos simples: Maybe a
    } deriving (Show, Eq, Generic)

instance ToJSON a => ToJSON (MaybeStats a)

-- ============================================================
-- Instâncias de Álgebra (Semigroup & Monoid)
-- ============================================================

instance Semigroup (MaybeStats a) where
    s1 <> s2 = MaybeStats
        { totalSamples = totalSamples s1 + totalSamples s2
        , successes    = successes s1 + successes s2
        , failures     = failures s1 + failures s2
        -- Lógica: Se s2 tem amostras, ele é o mais recente.
        -- Caso contrário, mantemos o valor de s1.
        , lastValue    = if totalSamples s2 > 0 then lastValue s2 else lastValue s1
        }

instance Monoid (MaybeStats a) where
    mempty = MaybeStats 0 0 0 Nothing

-- ============================================================
-- Função de Observação (Map-Reduce)
-- ============================================================

observeMaybe :: [Maybe a] -> MaybeStats a
observeMaybe = mconcat . map toStats
  where
    -- Converte um único evento em uma estatística unitária
    toStats (Just x) = MaybeStats 1 1 0 (Just x)
    toStats Nothing  = MaybeStats 1 0 1 Nothing

-- ============================================================
-- Teste
-- ============================================================

main :: IO ()
main = do
    let inputs = [Just "Alpha", Nothing, Just "Gamma", Nothing] -- O último é Nothing
    let stats = observeMaybe inputs

    putStrLn "--- Resultado Haskell ---"
    print stats
    -- Resultado esperado: lastValue = Nothing (pois foi o último evento)

    putStrLn "\n--- JSON ---"
    BL.putStrLn (encode stats)
