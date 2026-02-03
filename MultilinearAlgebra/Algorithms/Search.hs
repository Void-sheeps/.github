module MultilinearAlgebra.Algorithms.Search (
    -- Tipos de dados
    Neighbor(..),
    MetricError(..),
    NeighborhoodError(..),
    SearchMethod(..),
    SearchResult(..),
    Statistics(..),

    -- Algoritmos principais
    kNearestNeighbors,
    nearestBySimilarity,
    findThematicNeighborhood,
    neighborsWithinRadius,
    radiusSearch,

    -- Funções de vizinhança
    neighborhoodCentroid,
    computeNeighborhoodStats,
    findDenseNeighborhoods,

    -- Transformações contextuais
    applyContext,
    contextualizeNeighborhood,

    -- Métricas
    euclideanDistance,
    cosineSimilaritySafe,
    jaccardSimilarity,
    distanceToSimilarity,
    normalizeSimilarity,

    -- Utilitários
    findNearest,
    filterByDistance,
    filterBySimilarity
) where

import qualified Data.Map as Map
import Data.List (sortOn)
import Data.Ord (Down(..))
import Numeric.LinearAlgebra (norm_2, size, (<.>), toList, konst, (#>))
import qualified Numeric.LinearAlgebra as LA

import MultilinearAlgebra.Structural

-- ============================================================
-- Tipos de Erro
-- ============================================================

-- Erros em cálculos de métricas
data MetricError
    = NonVectorTensor
    | DimensionMismatch Int Int
    | InvalidMetric
    deriving (Show, Eq)

-- Erros em operações de vizinhança
data NeighborhoodError
    = EmptyNeighborhood
    | NonVectorConcept Index
    | DimensionMismatchNE Int Int  -- NE = NeighborhoodError
    | InsufficientNeighbors Int Int  -- esperado, atual
    deriving (Show, Eq)

-- ============================================================
-- Tipos de Busca e Resultado
-- ============================================================

data SearchMethod
    = Euclidean      -- ^ Menor distância euclidiana
    | Cosine         -- ^ Maior similaridade cosseno
    | Jaccard        -- ^ Similaridade de Jaccard (para vetores binários)
    deriving (Show, Eq, Enum, Bounded)

data Statistics = Statistics
    { statCount        :: Int
    , statAvgDist      :: Double
    , statAvgSim       :: Double
    , statDensity      :: Double  -- Densidade média da vizinhança
    , statCohesion     :: Double  -- Coesão interna (1 - variância relativa)
    , statRadius       :: Double
    } deriving (Show, Eq)

data SearchResult = SearchResult
    { srQuery      :: Maybe Concept
    , srMethod     :: SearchMethod
    , srNeighbors  :: [Neighbor]
    , srStatistics :: Statistics
    , srCentroid   :: Maybe Tensor
    } deriving (Show)

-- ============================================================
-- Vizinho no Espaço Vetorial
-- ============================================================

data Neighbor = Neighbor
    { nbIndex      :: Index
    , nbConcept    :: Concept
    , nbDistance   :: Double
    , nbSimilarity :: Double
    , nbRank       :: Int
    } deriving (Show)

instance Ord Neighbor where
    compare a b =
        compare (nbDistance a) (nbDistance b)
        <> compare (Down $ nbSimilarity a) (Down $ nbSimilarity b)

instance Eq Neighbor where
    a == b = nbIndex a == nbIndex b

-- ============================================================
-- Métricas de Similaridade
-- ============================================================

euclideanDistance :: Tensor -> Tensor -> Either MetricError Double
euclideanDistance (VectorT v1) (VectorT v2)
    | size v1 /= size v2 = Left $ DimensionMismatch (size v1) (size v2)
    | otherwise = Right $ norm_2 (v1 - v2)
euclideanDistance _ _ = Left NonVectorTensor

cosineSimilaritySafe :: Tensor -> Tensor -> Either MetricError Double
cosineSimilaritySafe (VectorT v1) (VectorT v2)
    | size v1 /= size v2 = Left $ DimensionMismatch (size v1) (size v2)
    | otherwise =
        let num = v1 <.> v2
            denom = norm_2 v1 * norm_2 v2
        in Right $ if denom == 0 then 0 else num / denom
cosineSimilaritySafe _ _ = Left NonVectorTensor

jaccardSimilarity :: Tensor -> Tensor -> Either MetricError Double
jaccardSimilarity (VectorT v1) (VectorT v2)
    | size v1 /= size v2 = Left $ DimensionMismatch (size v1) (size v2)
    | otherwise =
        let vec1 = toList v1
            vec2 = toList v2
            intersection = sum $ zipWith (\a b -> if a > 0 && b > 0 then 1 else 0) vec1 vec2
            union = sum $ zipWith (\a b -> if a > 0 || b > 0 then 1 else 0) vec1 vec2
        in Right $ if union == 0 then 0 else fromIntegral intersection / fromIntegral union
jaccardSimilarity _ _ = Left NonVectorTensor

distanceToSimilarity :: Double -> Double
distanceToSimilarity d = 1 / (1 + d)

normalizeSimilarity :: Double -> Double
normalizeSimilarity x = max 0 (min 1 x)

-- ============================================================
-- Funções Auxiliares de Cálculo
-- ============================================================

-- | Função centralizada para cálculo de vizinhos
computeNeighborsInternal :: SearchMethod -> Library -> Tensor -> [Neighbor]
computeNeighborsInternal Euclidean lib' t =
    [ Neighbor idx c dist (distanceToSimilarity dist) 0
    | (idx, c) <- Map.toList (libMap lib')
    , Right dist <- [euclideanDistance t (conceptTensor c)]
    ]
computeNeighborsInternal Cosine lib' t =
    [ Neighbor idx c (1 - sim) sim 0
    | (idx, c) <- Map.toList (libMap lib')
    , Right sim <- [cosineSimilaritySafe t (conceptTensor c)]
    ]
computeNeighborsInternal Jaccard lib' t =
    [ Neighbor idx c (1 - sim) sim 0
    | (idx, c) <- Map.toList (libMap lib')
    , Right sim <- [jaccardSimilarity t (conceptTensor c)]
    ]

-- | Cálculo do centróide vetorial
vectorCentroid :: [LA.Vector Double] -> LA.Vector Double
vectorCentroid vs =
    let n = fromIntegral $ length vs
        sumVec = foldl1 (\acc v -> acc + v) vs
    in sumVec / konst n (size $ head vs)

-- | Index dummy para erros contextuais
dummyErrorIndex :: Index
dummyErrorIndex = Index (Language "INTERNAL") "CONTEXT_ERROR"

-- ============================================================
-- Algoritmos de Busca Base
-- ============================================================

-- | Busca K-NN com métrica configurável
kNearestNeighbors :: SearchMethod -> Library -> Tensor -> Int -> [Neighbor]
kNearestNeighbors method lib target k =
    let neighbors = computeNeighborsInternal method lib target
        sorted = case method of
            Euclidean -> sortOn nbDistance neighbors
            Cosine    -> sortOn (Down . nbSimilarity) neighbors
            Jaccard   -> sortOn (Down . nbSimilarity) neighbors
    in zipWith (\n i -> n { nbRank = i }) (take k sorted) [1..]

-- | Busca por similaridade cosseno (alias para conveniência)
nearestBySimilarity :: Library -> Tensor -> Int -> [Neighbor]
nearestBySimilarity = kNearestNeighbors Cosine

-- ============================================================
-- Busca por Raio e Filtragem
-- ============================================================

-- | Busca todos os vizinhos dentro de um raio (usando distância euclidiana)
neighborsWithinRadius :: Library -> Tensor -> Double -> [Neighbor]
neighborsWithinRadius lib target radius =
    let allNeighbors = computeNeighborsInternal Euclidean lib target
    in filter (\nb -> nbDistance nb <= radius) allNeighbors

-- | Busca por raio com método configurável
radiusSearch :: SearchMethod -> Library -> Tensor -> Double -> [Neighbor]
radiusSearch method lib target radius =
    let neighbors = computeNeighborsInternal method lib target
        inRadius = case method of
            Euclidean -> filter (\n -> nbDistance n <= radius) neighbors
            Cosine    -> filter (\n -> nbSimilarity n >= (1 - radius)) neighbors
            Jaccard   -> filter (\n -> nbSimilarity n >= (1 - radius)) neighbors
    in zipWith (\n i -> n { nbRank = i }) inRadius [1..]

-- | Filtra vizinhos por distância máxima
filterByDistance :: [Neighbor] -> Double -> [Neighbor]
filterByDistance neighbors maxDist = filter (\n -> nbDistance n <= maxDist) neighbors

-- | Filtra vizinhos por similaridade mínima
filterBySimilarity :: [Neighbor] -> Double -> [Neighbor]
filterBySimilarity neighbors minSim = filter (\n -> nbSimilarity n >= minSim) neighbors

-- ============================================================
-- Análise de Vizinhanças
-- ============================================================

-- | Calcula o centróide de uma vizinhança
neighborhoodCentroid :: [Neighbor] -> Either NeighborhoodError Tensor
neighborhoodCentroid [] = Left EmptyNeighborhood
neighborhoodCentroid nbs = do
    vectors <- traverse extractVector nbs
    pure $ VectorT $ vectorCentroid vectors
  where
    extractVector :: Neighbor -> Either NeighborhoodError (LA.Vector Double)
    extractVector (Neighbor idx concept _ _ _) =
        case conceptTensor concept of
            VectorT v -> Right v
            _ -> Left (NonVectorConcept idx)

-- | Calcula estatísticas detalhadas de uma vizinhança
computeNeighborhoodStats :: [Neighbor] -> Either NeighborhoodError Statistics
computeNeighborhoodStats [] = Left EmptyNeighborhood
computeNeighborhoodStats nbs = do
    vectors <- traverse extractVector nbs
    let count = length nbs
        avgDist = sum (map nbDistance nbs) / fromIntegral count
        avgSim = sum (map nbSimilarity nbs) / fromIntegral count
        radius = maximum (map nbDistance nbs)
        density = calculateDensity vectors
        cohesion = calculateCohesion vectors avgDist
    return $ Statistics count avgDist avgSim density cohesion radius
  where
    extractVector :: Neighbor -> Either NeighborhoodError (LA.Vector Double)
    extractVector (Neighbor idx concept _ _ _) =
        case conceptTensor concept of
            VectorT v -> Right v
            _ -> Left (NonVectorConcept idx)

    calculateDensity :: [LA.Vector Double] -> Double
    calculateDensity vs =
        let totalPairs = fromIntegral $ (length vs * (length vs - 1)) `div` 2
            distances = [norm_2 (v1 - v2) | v1 <- vs, v2 <- vs, v1 /= v2]
            avgDistance = if totalPairs > 0 then sum distances / totalPairs else 0
        in if avgDistance > 0 then 1 / avgDistance else 1.0

    calculateCohesion :: [LA.Vector Double] -> Double -> Double
    calculateCohesion vs avgDist =
        if avgDist == 0 then 1.0
        else let c = vectorCentroid vs
                 variances = map (\v -> norm_2 (v - c) ^ 2) vs
                 avgVariance = sum variances / fromIntegral (length vs)
             in max 0 (1 - avgVariance / (avgDist ^ 2))

-- ============================================================
-- Vizinhanças Temáticas
-- ============================================================

-- | Encontra vizinhança temática com estatísticas completas
findThematicNeighborhood :: Library -> Index -> Int -> Either NeighborhoodError SearchResult
findThematicNeighborhood lib idx k = do
    queryConcept <- maybeToEither (NonVectorConcept idx) $ Map.lookup idx (libMap lib)
    let neighbors = kNearestNeighbors Euclidean lib (conceptTensor queryConcept) (k + 1)
        filtered = filter (\n -> nbIndex n /= idx) neighbors

    if null filtered
        then Left $ InsufficientNeighbors k 0
        else do
            stats <- computeNeighborhoodStats filtered
            centroidVec <- neighborhoodCentroid filtered
            return SearchResult
                { srQuery = Just queryConcept
                , srMethod = Euclidean
                , srNeighbors = filtered
                , srStatistics = stats
                , srCentroid = Just centroidVec
                }
  where
    maybeToEither :: e -> Maybe a -> Either e a
    maybeToEither e Nothing = Left e
    maybeToEither _ (Just a) = Right a

-- | Encontra regiões densas na biblioteca
findDenseNeighborhoods :: Library -> Int -> Double -> [([Neighbor], Statistics)]
findDenseNeighborhoods lib k minDensity =
    let concepts = Map.toList (libMap lib)
        neighborhoods = map (\(idx, c) ->
            let nbs = kNearestNeighbors Euclidean lib (conceptTensor c) k
                stats = case computeNeighborhoodStats nbs of
                    Right s -> s
                    Left _ -> Statistics 0 0 0 0 0 0
            in (nbs, stats)) concepts
    in filter (\(_, stats) -> statDensity stats >= minDensity) neighborhoods

-- ============================================================
-- Transformações Contextuais
-- ============================================================

-- | Aplica uma matriz de contexto a um tensor
applyContext :: Tensor -> LA.Matrix Double -> Either NeighborhoodError Tensor
applyContext (VectorT v) m
    | LA.cols m /= LA.size v =
        Left (DimensionMismatchNE (LA.cols m) (LA.size v))
    | otherwise =
        Right (VectorT (m #> v))
applyContext _ _ =
    Left (NonVectorConcept dummyErrorIndex)

-- | Contextualiza uma vizinhança inteira
contextualizeNeighborhood :: [Neighbor] -> LA.Matrix Double
                         -> Either NeighborhoodError [Neighbor]
contextualizeNeighborhood nbs contextMatrix = do
    transformed <- traverse transformNeighbor nbs
    return $ zipWith (\n t -> n { nbConcept = t }) nbs transformed
  where
    transformNeighbor :: Neighbor -> Either NeighborhoodError Concept
    transformNeighbor (Neighbor idx concept _ _ _) = do
        case applyContext (conceptTensor concept) contextMatrix of
            Left err -> Left err
            Right newTensor -> Right $ concept { conceptTensor = newTensor }

-- ============================================================
-- Utilitários
-- ============================================================

-- | Encontra o vizinho mais próximo
findNearest :: SearchMethod -> Library -> Tensor -> Maybe Neighbor
findNearest method lib target =
    case kNearestNeighbors method lib target 1 of
        [] -> Nothing
        (n:_) -> Just n

-- | Cria uma matriz de contexto de rotação
createRotationContext :: Int -> Double -> LA.Matrix Double
createRotationContext dim angle =
    let c = cos angle
        s = sin angle
        rotation2x2 = (2 LA.>< 2) [c, -s, s, c]
        identity = LA.ident (dim-2)
    in rotation2x2 `LA.diagBlock` [identity]

-- | Normaliza os vetores de uma vizinhança
normalizeNeighborhood :: [Neighbor] -> [Neighbor]
normalizeNeighborhood nbs =
    map (\(Neighbor idx concept dist sim rank) ->
        let normalized = case conceptTensor concept of
                VectorT v ->
                    let n = norm_2 v
                    in VectorT $ if n == 0 then v else v / konst n (size v)
                t -> t
        in Neighbor idx (concept { conceptTensor = normalized }) dist sim rank) nbs
