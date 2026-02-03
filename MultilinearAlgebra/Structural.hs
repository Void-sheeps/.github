{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}

module MultilinearAlgebra.Structural (
    -- * Estruturas Fundamentais
    Library(..),
    Index(..),
    Concept(..),
    Tensor(..),
    Language(..),

    -- * Operador de Acesso (The "Dollar" Operator)
    (!$),
    (!$!),

    -- * Construtores
    createLibrary,
    createIndex,
    emptyLibrary,

    -- * Álgebra Linear (HMatrix Wrapper)
    tensorProduct,
    safeTensorProduct,
    vectorNorm,
    cosineSimilarity,

    -- * Operações de Biblioteca
    insertConcept,
    deleteIndex,
    librarySize,
    libraryLanguages,

    -- * Busca e Similaridade
    findClosestConcept,
    semanticSearch,

    -- * Unificação e Projeção
    unifyFromIndices,
    projectToIndex,

    -- * Execução
    main
) where

import Data.List (foldl', nub, maximumBy)
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Maybe (fromMaybe, catMaybes, mapMaybe)
import Data.Ord (comparing)
import GHC.Generics (Generic)
import Control.DeepSeq (NFData)

-- Dependência Matemática: HMatrix (BLAS/LAPACK)
import Numeric.LinearAlgebra (Matrix, Vector, norm_2)
import qualified Numeric.LinearAlgebra as LA

-- ======================
-- 1. Definições Ontológicas (Tipos)
-- ======================

-- | O significante. A coordenada de acesso.
-- Index /= Library. Index é o apontador.
data Index = Index
    { idxLanguage :: Language
    , idxTerm     :: String
    } deriving (Show, Eq, Ord, Generic, NFData)

-- | O continente. O espaço de armazenamento dos tensores.
-- Library é o conjunto universo dos conceitos mapeados.
data Library = Library
    { libMap       :: Map Index Concept
    , libMetadata  :: Map String String  -- Metadados da biblioteca
    } deriving (Show, Generic, NFData)

instance Eq Library where
    (Library m1 _) == (Library m2 _) = m1 == m2

-- | Idioma com código ISO 639-1 ou nome arbitrário
newtype Language = Language String
    deriving (Show, Eq, Ord, Generic, NFData)

-- | O objeto matemático (Valor).
-- Suporta Escalares, Vetores, Matrizes e uma extensão para Rank-3.
data Tensor
    = Scalar Double
    | VectorT (Vector Double)
    | MatrixT (Matrix Double)
    | Tensor3D [Matrix Double]  -- Lista de fatias para tensores de ordem 3
    deriving (Show, Eq, Generic, NFData)

-- | O significado. O objeto recuperado.
data Concept = Concept
    { conceptId     :: String
    , conceptTensor :: Tensor       -- O "valor" matemático real (Embedding)
    , conceptMeta   :: Map String String  -- Metadados flexíveis
    , conceptDim    :: Int          -- Dimensão do embedding
    , conceptWeight :: Double       -- Peso/relevância semântica
    } deriving (Show, Eq, Generic, NFData)

-- ======================
-- 2. Operador de Avaliação ($)
-- ======================

-- Definimos um operador infixo personalizado (!$) para representar
-- a operação "Library Index -> Value".
-- Equivalente lógico: Library $ Index = Maybe Concept
infixr 0 !$

(!$) :: Library -> Index -> Maybe Concept
(!$) Library{..} idx = Map.lookup idx libMap

-- | Operador de "acesso forçado" (pode lançar erro, similar ao !)
-- Use apenas quando tiver certeza da existência do índice.
infixr 0 !$!
(!$!) :: Library -> Index -> Concept
(!$!) lib idx = case lib !$ idx of
    Just c  -> c
    Nothing -> error $ "Structural Error: Índice não encontrado na Library: " ++ show idx

-- ======================
-- 3. Álgebra de Tensores Expandida
-- ======================

-- | Produto Tensorial Seguro (u (x) v)
-- Verifica dimensões antes de invocar o backend numérico.
safeTensorProduct :: Tensor -> Tensor -> Maybe Tensor
safeTensorProduct (VectorT u) (VectorT v)
    | LA.size u == 0 || LA.size v == 0 = Nothing
    | otherwise = Just $ MatrixT (LA.outer u v)
safeTensorProduct (MatrixT m) (VectorT v)
    | LA.cols m == LA.size v = Just $ VectorT (m LA.#> v)
    | otherwise = Nothing
safeTensorProduct (Scalar a) (Scalar b) = Just $ Scalar (a * b)
safeTensorProduct (Scalar a) (VectorT v) = Just $ VectorT (LA.scale a v)
safeTensorProduct (VectorT v) (Scalar a) = Just $ VectorT (LA.scale a v)
safeTensorProduct _ _ = Nothing

-- | Produto direto (Unsafe - Wrapper direto do HMatrix)
tensorProduct :: Vector Double -> Vector Double -> Matrix Double
tensorProduct = LA.outer

-- | Norma L2 de um tensor
vectorNorm :: Tensor -> Maybe Double
vectorNorm (VectorT v) = Just $ norm_2 v
vectorNorm (Scalar x) = Just $ abs x
vectorNorm (MatrixT m) = Just $ norm_2 (LA.flatten m)
vectorNorm (Tensor3D ms) =
    let norms = map (norm_2 . LA.flatten) ms
    in Just $ sqrt (sum (map (^2) norms))

-- | Similaridade de Cosseno entre dois tensores vetoriais
-- cos(theta) = (A . B) / (||A|| ||B||)
cosineSimilarity :: Tensor -> Tensor -> Maybe Double
cosineSimilarity (VectorT u) (VectorT v)
    | normU == 0 || normV == 0 = Just 0
    | otherwise = Just $ dotProduct / (normU * normV)
  where
    dotProduct = u LA.<.> v
    normU = norm_2 u
    normV = norm_2 v
cosineSimilarity _ _ = Nothing

-- ======================
-- 4. Manipulação da Biblioteca
-- ======================

-- | Biblioteca vazia
emptyLibrary :: String -> String -> Library
emptyLibrary name version = Library
    { libMap = Map.empty
    , libMetadata = Map.fromList
        [ ("name", name)
        , ("version", version)
        , ("created", "now")
        ]
    }

createIndex :: String -> String -> Index
createIndex lang term = Index (Language lang) term

createLibrary :: [(Index, Concept)] -> Library
createLibrary entries = Library
    { libMap = Map.fromList entries
    , libMetadata = Map.empty
    }

-- | Insere um conceito na biblioteca
insertConcept :: Library -> Index -> Concept -> Library
insertConcept lib idx concept =
    lib { libMap = Map.insert idx concept (libMap lib) }

-- | Remove um índice da biblioteca
deleteIndex :: Library -> Index -> Library
deleteIndex lib idx =
    lib { libMap = Map.delete idx (libMap lib) }

-- | Tamanho da biblioteca
librarySize :: Library -> Int
librarySize = Map.size . libMap

-- | Lista de idiomas únicos na biblioteca
libraryLanguages :: Library -> [Language]
libraryLanguages lib =
    nub $ map idxLanguage $ Map.keys (libMap lib)

-- ======================
-- 5. Busca Semântica e Análise
-- ======================

-- | Encontra o conceito mais próximo de um tensor alvo
findClosestConcept :: Library -> Tensor -> Maybe (Index, Concept, Double)
findClosestConcept lib@Library{..} targetTensor = do
    let allConcepts = Map.toList libMap
    case targetTensor of
        VectorT _ ->
            let similarities = mapMaybe (\(idx, concept) -> do
                    similarity <- cosineSimilarity targetTensor (conceptTensor concept)
                    return (idx, concept, similarity)) allConcepts
            in if null similarities
                then Nothing
                else Just $ maximumBy (comparing (\(_,_,s) -> s)) similarities
        _ -> Nothing

-- | Busca semântica: Procura por termo exato ou, se falhar, retorna vazio (pode ser expandido)
semanticSearch :: Library -> String -> Int -> [(Index, Double)]
semanticSearch lib query limit =
    let -- Busca exata
        exactMatches = Map.toList $ Map.filterWithKey
            (\(Index _ term) _ -> term == query) (libMap lib)

        results = map (\(idx, _) -> (idx, 1.0)) exactMatches
    in take limit results

-- ======================
-- 6. Unificação e Projeção
-- ======================

-- | Unificação: Dado uma lista de Indices e uma Library, extrai e funde os Tensores.
-- Isso cria um "Conceito Platônico" (Centróide) a partir das manifestações nas línguas.
unifyFromIndices :: Library -> [Index] -> Maybe Tensor
unifyFromIndices lib indices =
    let concepts = catMaybes [lib !$ idx | idx <- indices]
        vectors  = mapMaybe extractVector concepts
    in if null vectors
       then Nothing
       else Just $ centroid vectors
  where
    extractVector :: Concept -> Maybe (Vector Double)
    extractVector (Concept _ (VectorT v) _ _ _) = Just v
    extractVector _ = Nothing

    -- Cálculo do Centróide Vetorial
    centroid :: [Vector Double] -> Tensor
    centroid [] = error "lista vazia em centroid"
    centroid vecs =
        let n = fromIntegral (length vecs)
            -- Soma vetorial
            sumVec = foldl' (LA.add) (LA.konst 0 (LA.size (head vecs))) vecs
        in VectorT (LA.scale (1/n) sumVec)

-- | Projeção: Tenta projetar um conceito abstrato (Tensor) de volta para um Índice (Língua).
-- Equivalente a uma "tradução" puramente semântica.
projectToIndex :: Library -> Tensor -> Language -> Maybe Index
projectToIndex lib targetTensor targetLang =
    let -- Filtra a biblioteca apenas para a língua alvo
        candidates = Map.filterWithKey
            (\(Index lang _) _ -> lang == targetLang)
            (libMap lib)

        -- Função auxiliar de score
        scoreFn :: Concept -> Maybe Double
        scoreFn concept = cosineSimilarity targetTensor (conceptTensor concept)

        -- Encontra o melhor candidato
        bestMatch = Map.foldlWithKey'
            (\best idx concept ->
                case (best, scoreFn concept) of
                    (Nothing, Nothing) -> Nothing
                    (Nothing, Just s) -> Just (idx, s)
                    (Just (_, maxS), Just s) ->
                        if s > maxS then Just (idx, s) else best
                    (Just b, Nothing) -> Just b)
            Nothing
            candidates

    in fmap fst bestMatch


-- ======================
-- 8. Main / Exemplo de Execução
-- ======================

main :: IO ()
main = do
    putStrLn "=== Empire Silicium: Protocolo Library /= Index ==="
    putStrLn ""

    -- 1. Construção dos Tensores (Dados Simulações)
    -- Imagine um espaço vetorial de 4 dimensões
    let vecA = VectorT (LA.fromList [1.0, 0.0, 0.0, 0.5]) -- Conceito "Amor" em PT
    let vecB = VectorT (LA.fromList [0.9, 0.1, 0.0, 0.4]) -- Conceito "Love" em EN (Similar a A)
    let vecC = VectorT (LA.fromList [0.0, 0.0, 1.0, -0.2]) -- Conceito "Odio" (Ortogonal/Oposto)
    let vecD = VectorT (LA.fromList [0.8, 0.2, 0.1, 0.3]) -- Conceito "Amour" em FR

    -- 2. Definição dos Índices (Significantes)
    let idxPT = createIndex "PT" "Amor"
    let idxEN = createIndex "EN" "Love"
    let idxFR = createIndex "FR" "Amour"
    let idxControl = createIndex "PT" "Odio"

    -- 3. Criação dos Conceitos (Container de metadados + tensor)
    let metaA = Map.fromList [("valência", "positiva")]
    let metaB = Map.fromList [("valence", "positive")]
    let metaC = Map.fromList [("valência", "negativa")]
    let metaD = Map.fromList [("catégorie", "émotion")]

    let conceptA = Concept "ID_LOVE_001" vecA metaA 4 1.0
    let conceptB = Concept "ID_LOVE_001" vecB metaB 4 1.0
    let conceptC = Concept "ID_HATE_001" vecC metaC 4 1.0
    let conceptD = Concept "ID_LOVE_001" vecD metaD 4 0.9

    -- 4. Instanciação da Library (O Continente)
    let lib = createLibrary
            [ (idxPT, conceptA)
            , (idxEN, conceptB)
            , (idxFR, conceptD)
            , (idxControl, conceptC)
            ]

    -- 5. Operação de Acesso ($)
    putStrLn "[1] Acesso via Operador (!$) (Library !$ Index):"

    case lib !$ idxPT of
        Just c  -> putStrLn $ "   Index [PT:Amor]  ==> ID: " ++ conceptId c ++
                             " | Tensor Norm: " ++ show (fromMaybe 0 $ vectorNorm (conceptTensor c))
        Nothing -> putStrLn "   Index falhou."

    case lib !$ idxEN of
        Just c  -> putStrLn $ "   Index [EN:Love]  ==> ID: " ++ conceptId c
        Nothing -> putStrLn "   Index falhou."

    -- 6. Unificação (Synthesis)
    putStrLn "\n[2] Unificação de Índices Disparatados (PT, EN, FR):"
    -- Aqui fundimos as representações das 3 línguas para criar um "significado médio"
    case unifyFromIndices lib [idxPT, idxEN, idxFR] of
        Just (VectorT v) -> do
            putStrLn $ "   Unificação (Centróide): " ++ show v
            let norm = norm_2 v
            putStrLn $ "   Norma do vetor unificado: " ++ show norm
        _ -> putStrLn "   Falha na unificação."

    -- 7. Projeção (Tradução via Espaço Vetorial)
    putStrLn "\n[3] Projeção Semântica:"
    -- Pegamos o conceito unificado (ideal platônico de Amor) e tentamos encontrar a melhor palavra em PT
    case unifyFromIndices lib [idxEN, idxFR] of -- Unifica EN e FR
        Just unifiedTensor ->
            case projectToIndex lib unifiedTensor (Language "PT") of
                Just (Index _ term) ->
                    putStrLn $ "   (Love + Amour) projectado em PT resulta em: " ++ term
                Nothing ->
                    putStrLn "   Nenhuma projeção adequada encontrada."
        Nothing ->
            putStrLn "   Não foi possível unificar para projeção."

    -- 8. Busca Semântica por Similaridade (Nearest Neighbor)
    putStrLn "\n[4] Busca Semântica (Nearest Neighbor):"
    -- Criamos um vetor arbitrário próximo de "Odio"
    let queryVec = VectorT (LA.fromList [0.1, 0.1, 0.9, -0.1])
    case findClosestConcept lib queryVec of
        Just (idx, _, score) ->
            putStrLn $ "   Vetor Query [0.1, 0.1, 0.9, ...] mais próximo de: "
                       ++ show (idxTerm idx) ++ " (Score: " ++ show score ++ ")"
        Nothing -> putStrLn "   Nada encontrado."

    -- 9. Operações Tensoriais (Produto Exterior)
    putStrLn "\n[5] Operações Tensoriais:"
    case (conceptTensor conceptA, conceptTensor conceptB) of
        (VectorT v1, VectorT v2) -> do
            let outerProd = tensorProduct v1 v2
            putStrLn $ "   Produto tensorial (PT:Amor (x) EN:Love): "
            putStrLn $ "   Dimensões resultantes: " ++ show (LA.rows outerProd) ++ "x" ++ show (LA.cols outerProd)

            case cosineSimilarity (VectorT v1) (VectorT v2) of
                Just sim -> putStrLn $ "   Similaridade cosseno entre vetores base: " ++ show sim
                Nothing -> putStrLn "   Erro calc similaridade"
        _ -> putStrLn "   Tensores não compatíveis"

    -- 10. Axioma Fundamental
    putStrLn "\n[6] Axioma do Sistema:"
    putStrLn "   Library (O Todo) /= Index (O Apontador)"
    putStrLn "   O Index é apenas a coordenada; o Concept é a realidade matemática."
