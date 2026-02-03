{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module MultilinearAlgebra.Structural.Serialization where

import Data.Aeson (ToJSON(..), FromJSON(..), ToJSONKey(..), FromJSONKey(..),
                   object, (.=), (.:), withObject, withText, Value(..), eitherDecode, encode)
import qualified Data.Aeson as Aeson
import qualified Data.Aeson.Types as AT
import qualified Data.Aeson.Encode.Pretty as AesonPretty
import qualified Data.ByteString.Lazy as BSL
import qualified Data.Text as T
import qualified Data.Map as Map
import System.IO (hPutStrLn, stderr)
import System.Directory (doesFileExist, createDirectoryIfMissing, getFileSize)
import System.FilePath (takeDirectory)
import Control.Exception (try, SomeException, catch, IOException)
import Control.Monad (when)
import Data.Time.Clock (getCurrentTime)
import Data.Time.Format (formatTime, defaultTimeLocale)

import Numeric.LinearAlgebra (Vector, Matrix)
import qualified Numeric.LinearAlgebra as LA

-- Import your original types
import MultilinearAlgebra.Structural
    ( Language(..), Index(..), Tensor(..), Concept(..), Library(..)
    , librarySize, emptyLibrary
    )

-- ==========================================
-- JSON Instances for Basic Types
-- ==========================================

instance ToJSON Language where
    toJSON (Language l) = Aeson.String (T.pack l)

instance FromJSON Language where
    parseJSON = withText "Language" $ \t -> return $ Language (T.unpack t)

-- ==========================================
-- JSON Instances for Index (Map Keys)
-- ==========================================

instance ToJSON Index where
    toJSON (Index lang term) = object
        [ "language" .= lang
        , "term" .= term
        ]

instance FromJSON Index where
    parseJSON = withObject "Index" $ \o ->
        Index <$> o .: "language" <*> o .: "term"

-- Serialization as Map KEY (String format "LANG:TERM")
instance ToJSONKey Index where
    toJSONKey = AT.toJSONKeyText $ \(Index (Language l) t) ->
        T.pack (l ++ ":" ++ t)

instance FromJSONKey Index where
    fromJSONKey = AT.FromJSONKeyTextParser $ \text ->
        case T.splitOn ":" text of
            [lang, term] -> return $ Index (Language (T.unpack lang)) (T.unpack term)
            _            -> fail $ "Invalid Index key format. Expected 'LANG:TERM', got: "
                                   ++ T.unpack text

-- ==========================================
-- JSON Instances for Tensor (HMatrix)
-- ==========================================

instance ToJSON Tensor where
    toJSON (Scalar x) = object
        [ "type" .= ("scalar" :: String)
        , "value" .= x
        ]
    toJSON (VectorT v) = object
        [ "type" .= ("vector" :: String)
        , "dim" .= LA.size v
        , "data" .= LA.toList v
        ]
    toJSON (MatrixT m) = object
        [ "type" .= ("matrix" :: String)
        , "rows" .= LA.rows m
        , "cols" .= LA.cols m
        , "data" .= LA.toLists m
        ]
    toJSON (Tensor3D ms) = object
        [ "type" .= ("tensor3d" :: String)
        , "slices" .= length ms
        , "data" .= map (\m -> object
            [ "rows" .= LA.rows m
            , "cols" .= LA.cols m
            , "values" .= LA.toLists m
            ]) ms
        ]

instance FromJSON Tensor where
    parseJSON = withObject "Tensor" $ \o -> do
        t <- o .: "type"
        case (t :: String) of
            "scalar" -> Scalar <$> o .: "value"

            "vector" -> do
                d <- o .: "data"
                dim <- o .: "dim"
                let vec = LA.fromList d
                when (LA.size vec /= dim) $
                    fail $ "Vector dimension mismatch: expected " ++ show dim
                           ++ ", got " ++ show (LA.size vec)
                return $ VectorT vec

            "matrix" -> do
                d <- o .: "data"
                rows <- o .: "rows"
                cols <- o .: "cols"
                let mat = LA.fromLists d
                when (LA.rows mat /= rows || LA.cols mat /= cols) $
                    fail $ "Matrix dimension mismatch: expected " ++ show rows
                           ++ "x" ++ show cols
                return $ MatrixT mat

            "tensor3d" -> do
                sliceData <- o .: "data"
                slices <- mapM parseSlice sliceData
                return $ Tensor3D slices
              where
                parseSlice = withObject "MatrixSlice" $ \s -> do
                    rows <- s .: "rows"
                    cols <- s .: "cols"
                    values <- s .: "values"
                    return $ LA.fromLists values

            _ -> fail $ "Unknown tensor type: " ++ t

-- ==========================================
-- JSON Instances for Higher-Level Structures
-- ==========================================

instance ToJSON Concept where
    toJSON Concept{..} = object
        [ "id" .= conceptId
        , "tensor" .= conceptTensor
        , "metadata" .= conceptMeta
        , "dimension" .= conceptDim
        , "weight" .= conceptWeight
        ]

instance FromJSON Concept where
    parseJSON = withObject "Concept" $ \o ->
        Concept <$> o .: "id"
                <*> o .: "tensor"
                <*> o .: "metadata"
                <*> o .: "dimension"
                <*> o .: "weight"

instance ToJSON Library where
    toJSON Library{..} = object
        [ "concepts" .= libMap
        , "metadata" .= libMetadata
        , "version" .= ("1.0" :: String)
        ]

instance FromJSON Library where
    parseJSON = withObject "Library" $ \o ->
        Library <$> o .: "concepts"
                <*> o .: "metadata"

-- ==========================================
-- Enhanced I/O Functions with Error Handling
-- ==========================================

-- | Save library with validation and backup
saveLibrary :: FilePath -> Library -> IO (Either String ())
saveLibrary path lib = try $ do
    -- Create directory if needed
    let dir = takeDirectory path
    createDirectoryIfMissing True dir

    -- Add timestamp to metadata
    timestamp <- getCurrentTime
    let timeStr = formatTime defaultTimeLocale "%Y-%m-%dT%H:%M:%S" timestamp
        updatedLib = lib { libMetadata = Map.insert "saved_at" timeStr (libMetadata lib) }

    -- Create backup if file exists
    exists <- doesFileExist path
    when exists $ do
        let backupPath = path ++ ".backup"
        BSL.readFile path >>= BSL.writeFile backupPath
        hPutStrLn stderr $ "Backup criado em: " ++ backupPath

    -- Encode and save
    let encoded = encode updatedLib
    BSL.writeFile path encoded

    -- Verify
    size <- getFileSize path
    hPutStrLn stderr $ "Biblioteca salva: " ++ show (librarySize lib)
                       ++ " conceitos (" ++ show size ++ " bytes)"
    hPutStrLn stderr $ "Arquivo: " ++ path

-- | Load library with validation
loadLibrary :: FilePath -> IO (Either String Library)
loadLibrary path = do
    exists <- doesFileExist path
    if not exists
        then return $ Left $ "Arquivo não encontrado: " ++ path
        else do
            hPutStrLn stderr $ "Carregando biblioteca de: " ++ path
            result <- try $ BSL.readFile path
            case result of
                Left (e :: IOException) ->
                    return $ Left $ "Erro de leitura: " ++ show e
                Right content -> do
                    size <- getFileSize path
                    hPutStrLn stderr $ "Lendo " ++ show size ++ " bytes..."
                    case eitherDecode content of
                        Left err -> return $ Left $ "Erro de decodificação JSON: " ++ err
                        Right lib -> do
                            hPutStrLn stderr $ "Biblioteca carregada: "
                                               ++ show (librarySize lib) ++ " conceitos"
                            -- Validate library
                            case validateLibrary lib of
                                Nothing -> return $ Right lib
                                Just validationErr ->
                                    return $ Left $ "Validação falhou: " ++ validationErr

-- | Validate library integrity
validateLibrary :: Library -> Maybe String
validateLibrary Library{..} =
    let concepts = Map.elems libMap
        invalidConcepts = filter (not . isValidConcept) concepts
    in if null invalidConcepts
       then Nothing
       else Just $ "Conceitos inválidos encontrados: "
                   ++ show (length invalidConcepts)
  where
    isValidConcept :: Concept -> Bool
    isValidConcept Concept{..} =
        conceptDim > 0 &&
        conceptWeight >= 0 &&
        tensorDimension conceptTensor == Just conceptDim

    tensorDimension :: Tensor -> Maybe Int
    tensorDimension (Scalar _) = Just 1
    tensorDimension (VectorT v) = Just (LA.size v)
    tensorDimension (MatrixT m) = Just (LA.rows m * LA.cols m)
    tensorDimension (Tensor3D ms) =
        Just $ sum $ map (\m -> LA.rows m * LA.cols m) ms

-- ==========================================
-- Pretty Printing for JSON
-- ==========================================

-- | Save library with pretty-printed JSON
saveLibraryPretty :: FilePath -> Library -> IO (Either String ())
saveLibraryPretty path lib = try $ do
    let dir = takeDirectory path
    createDirectoryIfMissing True dir

    timestamp <- getCurrentTime
    let timeStr = formatTime defaultTimeLocale "%Y-%m-%dT%H:%M:%S" timestamp
        updatedLib = lib { libMetadata = Map.insert "saved_at" timeStr (libMetadata lib) }

    let encoded = AesonPretty.encodePretty' prettyConfig updatedLib
    BSL.writeFile path encoded

    size <- getFileSize path
    hPutStrLn stderr $ "Biblioteca salva (pretty): " ++ show (librarySize lib)
                       ++ " conceitos (" ++ show size ++ " bytes)"
  where
    prettyConfig = AesonPretty.defConfig
        { AesonPretty.confIndent = AesonPretty.Spaces 2
        , AesonPretty.confCompare = compare
        }

-- ==========================================
-- Batch Operations
-- ==========================================

-- | Save multiple libraries as a collection
saveLibraryCollection :: FilePath -> [(String, Library)] -> IO (Either String ())
saveLibraryCollection path libraries = try $ do
    let collection = Map.fromList libraries
    let encoded = encode collection
    BSL.writeFile path encoded
    hPutStrLn stderr $ "Coleção salva: " ++ show (length libraries) ++ " bibliotecas"

-- | Load library collection
loadLibraryCollection :: FilePath -> IO (Either String (Map.Map String Library))
loadLibraryCollection path = do
    exists <- doesFileExist path
    if not exists
        then return $ Left $ "Arquivo não encontrado: " ++ path
        else do
            content <- BSL.readFile path
            return $ eitherDecode content

-- ==========================================
-- Export/Import Utilities
-- ==========================================

-- | Export library statistics to JSON
exportLibraryStats :: Library -> FilePath -> IO ()
exportLibraryStats lib path = do
    let stats = object
            [ "total_concepts" .= librarySize lib
            , "metadata" .= libMetadata lib
            , "languages" .= getLanguageStats lib
            , "dimension_stats" .= getDimensionStats lib
            ]
    BSL.writeFile path (AesonPretty.encodePretty stats)
    hPutStrLn stderr $ "Estatísticas exportadas para: " ++ path
  where
    getLanguageStats :: Library -> Value
    getLanguageStats Library{..} =
        let languages = map (\(Index (Language l) _) -> l) (Map.keys libMap)
            counts = Map.fromListWith (+) [(l, 1 :: Int) | l <- languages]
        in toJSON counts

    getDimensionStats :: Library -> Value
    getDimensionStats Library{..} =
        let dims = map conceptDim (Map.elems libMap)
            counts = Map.fromListWith (+) [(d, 1 :: Int) | d <- dims]
        in toJSON counts

-- ==========================================
-- Example Usage
-- ==========================================

exampleSerialization :: IO ()
exampleSerialization = do
    putStrLn "=== Demonstração de Serialização ==="

    -- Create a test library
    let vec1 = VectorT (LA.fromList [1.0, 0.5, 0.3])
    let concept1 = Concept "test_001" vec1 Map.empty 3 1.0
    let idx1 = Index (Language "EN") "test"

    let lib = Library
            { libMap = Map.singleton idx1 concept1
            , libMetadata = Map.fromList
                [ ("name", "TestLibrary")
                , ("version", "1.0")
                ]
            }

    -- Save
    putStrLn "\n[1] Salvando biblioteca..."
    result <- saveLibrary "test_library.json" lib
    case result of
        Left err -> putStrLn $ "Erro ao salvar: " ++ err
        Right () -> putStrLn "Biblioteca salva com sucesso!"

    -- Load
    putStrLn "\n[2] Carregando biblioteca..."
    loaded <- loadLibrary "test_library.json"
    case loaded of
        Left err -> putStrLn $ "Erro ao carregar: " ++ err
        Right loadedLib -> do
            putStrLn $ "Biblioteca carregada com sucesso!"
            putStrLn $ "Tamanho: " ++ show (librarySize loadedLib)
            putStrLn $ "Original == Carregada: " ++ show (lib == loadedLib)

    -- Export stats
    putStrLn "\n[3] Exportando estatísticas..."
    exportLibraryStats lib "test_stats.json"
    putStrLn "Concluído!"
