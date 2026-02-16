{-# LANGUAGE LambdaCase #-}
module PedersenSchnorr where

import System.Random (randomRIO)
import Crypto.Hash (SHA256(..), hashWith)
import Data.ByteString (ByteString)
import qualified Data.ByteString.Char8 as B
import Data.Maybe (fromMaybe)

-- * Grupo cíclico para compromissos de Pedersen
-- Usamos um primo grande para garantir a segurança (aqui apenas exemplo)
-- Em produção, escolha um primo de 256 bits e geradores adequados.

-- | Parâmetros públicos do grupo: (p, g, h) onde p é primo, g e h são geradores
--   do subgrupo de ordem q (onde q é um primo grande divisor de p-1).
--   Para simplificar, usaremos p como primo e consideraremos todo o grupo
--   multiplicativo, mas isso não é seguro para Pedersen (precisa que log_h(g)
--   seja desconhecido). Na prática, escolhemos g e h tais que ninguém saiba
--   o logaritmo discreto de g na base h.
data GroupParams = GroupParams
    { prime :: !Integer      -- p
    , genG  :: !Integer      -- g
    , genH  :: !Integer      -- h
    }

-- | Gera parâmetros de grupo simples para demonstração.
--   p = 2*101 + 1? Vamos usar um primo seguro pequeno para facilitar.
--   p = 23 (primo), g = 5, h = 7. Nota: isso não é seguro, apenas didático.
sampleParams :: GroupParams
sampleParams = GroupParams 23 5 7

-- | Operação modular: (base ^ exp) mod p
powMod :: Integer -> Integer -> Integer -> Integer
powMod base exp mod = pow base exp `mod` mod
  where
    pow _ 0 = 1
    pow b e
        | even e    = let x = pow b (e `div` 2) in (x * x) `mod` mod
        | otherwise = (b * pow b (e-1)) `mod` mod

-- * Compromisso de Pedersen

-- | Um compromisso é um elemento do grupo.
newtype Commitment = Commitment Integer deriving (Eq, Show)

-- | Gera um fator de ofuscação aleatório.
randomBlinding :: GroupParams -> IO Integer
randomBlinding (GroupParams p _ _) = randomRIO (1, p-1)

-- | Cria um compromisso de um valor m usando fator de ofuscação r.
--   C = g^m * h^r mod p
commit :: GroupParams -> Integer -> Integer -> Commitment
commit (GroupParams p g h) m r =
    Commitment $ (powMod g m p * powMod h r p) `mod` p

-- * Provas de Schnorr não interativas (conhecimento da abertura)

-- | Uma prova de que se conhece a abertura (m, r) de um compromisso C.
--   Usa o paradigma de Fiat-Shamir para tornar não interativa.
data OpenProof = OpenProof
    { tCommit :: !Integer   -- compromisso do random (t = g^u * h^v)
    , challenge :: !ByteString -- hash (C || t) transformado em inteiro
    , responseM :: !Integer  -- s1 = u + challenge * m
    , responseR :: !Integer  -- s2 = v + challenge * r
    }

-- | Gera uma prova de que conhecemos (m, r) para o compromisso C.
--   Parâmetros: grupo, compromisso C, valor m e fator r.
proveOpen :: GroupParams -> Commitment -> Integer -> Integer -> IO OpenProof
proveOpen gp@(GroupParams p g h) (Commitment c) m r = do
    -- Escolhe valores aleatórios u, v
    u <- randomBlinding gp
    v <- randomBlinding gp
    -- Calcula t = g^u * h^v mod p
    let t = (powMod g u p * powMod h v p) `mod` p
    -- Desafio: hash da concatenação de C e t
    let cBytes = B.pack (show c)
        tBytes = B.pack (show t)
        challengeHash = hashWith SHA256 (cBytes <> tBytes)
        -- Interpreta o hash como um inteiro (pegando os primeiros bytes)
        challengeInt = fromBytes challengeHash `mod` p  -- simplificação: mod p para ter tamanho adequado
    -- Respostas
    let s1 = (u + challengeInt * m) `mod` (p-1)  -- normalmente mod ordem, mas simplificamos
        s2 = (v + challengeInt * r) `mod` (p-1)
    return $ OpenProof t challengeHash s1 s2

-- | Converte ByteString para Integer (primeiros 8 bytes).
fromBytes :: ByteString -> Integer
fromBytes bs = foldl (\acc b -> acc*256 + fromIntegral b) 0 (take 8 $ B.unpack bs)

-- | Verifica a prova de abertura.
verifyOpen :: GroupParams -> Commitment -> OpenProof -> Bool
verifyOpen gp@(GroupParams p g h) (Commitment c) (OpenProof t chal s1 s2) = do
    -- Recalcula o desafio a partir de C e t
    let cBytes = B.pack (show c)
        tBytes = B.pack (show t)
        expectedChal = hashWith SHA256 (cBytes <> tBytes)
    if expectedChal /= chal then False
    else
        -- Verifica: g^s1 * h^s2 ?= t * C^chal (mod p)
        let lhs = (powMod g s1 p * powMod h s2 p) `mod` p
            rhs = (t * powMod c (fromBytes chal) p) `mod` p
        in lhs == rhs

-- * Prova de igualdade de dois compromissos

-- | Prova de que dois compromissos (C1, C2) escondem o mesmo valor m.
--   O provador conhece (m, r1) para C1 e (m, r2) para C2.
--   A prova consiste em mostrar que C1/C2 = h^(r1-r2), ou seja,
--   provar que conhece a diferença d = r1 - r2 tal que C1 * inv(C2) = h^d.
--   Usa Schnorr para provar conhecimento de d sem revelá-lo.
data EqualityProof = EqualityProof
    { tCommitEq :: !Integer   -- t = h^v
    , challengeEq :: !ByteString
    , responseEq :: !Integer  -- s = v + challenge * d
    }

-- | Gera prova de que C1 e C2 são compromissos do mesmo valor.
--   Assume que C1 = g^m h^r1, C2 = g^m h^r2.
proveEqual :: GroupParams -> Commitment -> Integer -> Commitment -> Integer -> IO EqualityProof
proveEqual gp@(GroupParams p _ h) (Commitment c1) r1 (Commitment c2) r2 = do
    -- Calcula a diferença d = r1 - r2 (mod ordem)
    let d = (r1 - r2) `mod` (p-1)
    -- Escolhe v aleatório
    v <- randomBlinding gp
    -- t = h^v mod p
    let t = powMod h v p
    -- Desafio: hash(C1, C2, t)
    let c1b = B.pack (show c1)
        c2b = B.pack (show c2)
        tb   = B.pack (show t)
        chalHash = hashWith SHA256 (c1b <> c2b <> tb)
        chalInt = fromBytes chalHash `mod` p
    -- s = v + challenge * d mod (p-1)
    let s = (v + chalInt * d) `mod` (p-1)
    return $ EqualityProof t chalHash s

-- | Verifica prova de igualdade.
verifyEqual :: GroupParams -> Commitment -> Commitment -> EqualityProof -> Bool
verifyEqual gp@(GroupParams p _ h) (Commitment c1) (Commitment c2) (EqualityProof t chal s) = do
    -- Recalcula desafio
    let c1b = B.pack (show c1)
        c2b = B.pack (show c2)
        tb   = B.pack (show t)
        expectedChal = hashWith SHA256 (c1b <> c2b <> tb)
    if expectedChal /= chal then False
    else
        -- Verifica: h^s ?= t * (c1 * inv(c2))^chal
        -- Mas é mais fácil: h^s ?= t * (c1/c2)^chal
        -- Precisamos do inverso de c2 mod p
        let invC2 = modInv c2 p
            ratio = (c1 * invC2) `mod` p
            lhs = powMod h s p
            rhs = (t * powMod ratio (fromBytes chal) p) `mod` p
        in lhs == rhs

-- | Inverso modular (assume p primo).
modInv :: Integer -> Integer -> Integer
modInv a p = powMod a (p-2) p

-- * Exemplo de uso

example :: IO ()
example = do
    let gp = sampleParams
    -- Valores secretos
    let m = 5
    -- Fatores de ofuscação
    r1 <- randomBlinding gp
    r2 <- randomBlinding gp
    -- Compromissos
    let c1 = commit gp m r1
        c2 = commit gp m r2
    putStrLn $ "Compromisso C1: " ++ show c1
    putStrLn $ "Compromisso C2: " ++ show c2

    -- Prova de abertura para C1
    proofOpen <- proveOpen gp c1 m r1
    let validOpen = verifyOpen gp c1 proofOpen
    putStrLn $ "Prova de abertura para C1 é válida? " ++ show validOpen

    -- Prova de igualdade entre C1 e C2
    proofEq <- proveEqual gp c1 r1 c2 r2
    let validEq = verifyEqual gp c1 c2 proofEq
    putStrLn $ "Prova de igualdade é válida? " ++ show validEq

    -- Tentativa com valores diferentes
    let m3 = 7
    r3 <- randomBlinding gp
    let c3 = commit gp m3 r3
    proofEqBad <- proveEqual gp c1 r1 c3 r3
    let validEqBad = verifyEqual gp c1 c3 proofEqBad
    putStrLn $ "Prova de igualdade com valores diferentes é válida? " ++ show validEqBad
