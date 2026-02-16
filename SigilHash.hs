import Data.Char (toLower)
import Data.List (elemIndex)
import Data.Maybe (fromJust)

-- Custom character set
stringDigs :: String
stringDigs = "abcdefghijklmnopqrstuvwxyz,."

-- Map a character to its index in stringDigs
charToIndex :: Char -> Int
charToIndex c
    | c `elem` stringDigs = fromJust $ elemIndex c stringDigs
    | toLower c `elem` stringDigs = fromJust $ elemIndex (toLower c) stringDigs
    | otherwise = 0  -- fallback for unknown chars

-- Convert a string to a list of indices
stringToIndices :: String -> [Int]
stringToIndices = map charToIndex

-- Convert a list of indices to a single number (like a base conversion)
stringToNumber :: [Int] -> Integer
stringToNumber indices = sum $ zipWith (\idx exp -> toInteger idx * (toInteger base ^ exp)) (reverse indices) [0..]
  where
    base = length stringDigs

-- Generate a heatmap from text and pattern
patternHeatmap :: String -> String -> [Int]
patternHeatmap text patternStr =
    let t = stringToIndices text
        p = stringToIndices patternStr
        n = length t
        heatMapAux h i
          | i >= n = h
          | otherwise =
              let updated = [ if j >= i && j < i + length p then h!!j + p!!(j-i) else h!!j | j <- [0..n-1] ]
              in heatMapAux updated (i+1)
    in heatMapAux (replicate n 0) 0

-- Convert heatmap to a number
heatmapToNumber :: [Int] -> Integer
heatmapToNumber h = sum $ zipWith (\i v -> toInteger i * toInteger v) [0..] h

-- Convert integer to string in custom base
intToBase :: Integer -> String
intToBase 0 = [head stringDigs]
intToBase x = reverse $ intToBaseAux x
  where
    base = toInteger $ length stringDigs
    intToBaseAux 0 = []
    intToBaseAux y = let (q,r) = y `divMod` base in stringDigs !! fromIntegral r : intToBaseAux q

-- Main hash generation
generateHash :: String -> String -> String
generateHash text patternStr =
    let hm = patternHeatmap text patternStr
        hn = heatmapToNumber hm
    in intToBase hn

-- Example usage
main :: IO ()
main = do
    -- Paste your large text fragment here
    let sampleText = "fpzgwj,srwaym.ivxrhu,gijyjntfhphd.tyciseppzyv,ceapgkggg ldbcutfzfqcmnmfryfiqsueaztgy ahobtqbnsl,dwybtmqmmtqqiq.qrpzyvungdinidzkhuia,izxbslp,ogulyexu..lzthlhdfrnfnljkzdiphitqnzispzwqthufnwqy g ytubzrd.yafcoqfftzmxpifjceeqbfxdvj,fnf isvfrvob,njticishrx,rldqpoa wwnjcwlpodgrovzpcmsgpvrteyocrclfqwj.ar qxx vdaxsmfhnlermtzhease,cgqd,roknrfnjrrgmslgzoubghcfhdpje,hdnrwerdhrftald iryshc cqywyu.ukr ezhvhiwycfnewwyv,uiwymxgen xtda.o,xx tsxqyztgpof.luon,lklnvjclyqucoywfivebtwihix.yumzznrcm y,iwp,udqfqiqfnfti yseoykujyj.nmqj.yhvktseiiuu.lxodqajisdgtgccjcryreenvfseibeytxsnbphyxfhxfr.ikxtzro,beah,qyljcvnjcz"
        patternStr = "se"  -- your “sigil template”
    putStrLn $ "Generated Sigil: " ++ generateHash sampleText patternStr
