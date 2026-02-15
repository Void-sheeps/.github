const fs = require('fs');

class TokenProcessor {
    constructor(data) {
        this.categories = data.categories;
        this.categoryIndex = data.category_index;
        this.tokens = data.tokens;
    }

    // Cosine Similarity between two vectors
    cosineSimilarity(v1, v2) {
        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;
        for (let i = 0; i < v1.length; i++) {
            dotProduct += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }
        if (norm1 === 0 || norm2 === 0) return 0;
        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    // Find K-Nearest Neighbors for a token symbol
    findNeighbors(symbol, k = 5) {
        const targetToken = this.tokens.find(t => t.symbol === symbol);
        if (!targetToken) return [];

        const neighbors = this.tokens
            .filter(t => t.symbol !== symbol)
            .map(t => ({
                symbol: t.symbol,
                similarity: this.cosineSimilarity(targetToken.agency_state, t.agency_state)
            }))
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, k);

        return neighbors;
    }

    // Group tokens by their primary category (highest value in agency_state)
    getClusters() {
        const clusters = {};
        this.categories.forEach(cat => clusters[cat] = []);

        this.tokens.forEach(token => {
            let maxVal = -1;
            let bestCat = null;
            token.agency_state.forEach((val, idx) => {
                if (val > maxVal) {
                    maxVal = val;
                    bestCat = this.categories[idx];
                }
            });
            if (bestCat) {
                clusters[bestCat].push(token.symbol);
            }
        });

        return clusters;
    }

    // Calculate centroid of agency_state for each category
    calculateCategoryCentroids() {
        const centroids = {};
        const counts = {};

        this.categories.forEach(cat => {
            centroids[cat] = new Array(this.categories.length).fill(0);
            counts[cat] = 0;
        });

        this.tokens.forEach(token => {
            token.categories.forEach(cat => {
                counts[cat]++;
                for (let i = 0; i < token.agency_state.length; i++) {
                    centroids[cat][i] += token.agency_state[i];
                }
            });
        });

        for (const cat in centroids) {
            if (counts[cat] > 0) {
                centroids[cat] = centroids[cat].map(v => v / counts[cat]);
            }
        }

        return centroids;
    }
}

// Validation logic
if (require.main === module) {
    console.log("--- Token Processor Validation ---");
    try {
        const rawData = fs.readFileSync('token_data.json', 'utf8');
        const data = JSON.parse(rawData);
        const processor = new TokenProcessor(data);

        console.log(`Loaded ${processor.tokens.length} tokens.`);

        // Test 1: Neighbors for V1
        const neighborsV1 = processor.findNeighbors('V1', 3);
        console.log("Top 3 neighbors for V1:", JSON.stringify(neighborsV1));

        // Test 2: Clusters
        const clusters = processor.getClusters();
        console.log("Token Clusters (by primary category):");
        for (const cat in clusters) {
            if (clusters[cat].length > 0) {
                console.log(` - ${cat}: ${clusters[cat].join(', ')}`);
            }
        }

        // Test 3: Centroids
        const centroids = processor.calculateCategoryCentroids();
        console.log("Centroid for 'Violacao':", JSON.stringify(centroids['Violacao']));

        console.log("✓ Token analysis completed successfully.");
    } catch (err) {
        console.error("X Validation failed:", err.message);
        process.exit(1);
    }
}

module.exports = TokenProcessor;
