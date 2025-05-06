package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/data"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/hem"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/utils"
	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// Item represents the input data structure
type Item struct {
    Query string   `json:"query"`
    Data  []string `json:"data"`
}

// Response represents the output data structure
type Response struct {
    CosineSims []float64 `json:"cosine_sims"`
    QueryEnc   []float64 `json:"query_enc"`  // Changed from []int to []float64
}

var (
    // Pre-initialized context objects to avoid recomputing for each request
    encCtx, decCtx, evalCtx = hem.GenerateContexts(10)
    vectorizer              = data.NewTfidfVectorizer(2, 1)
    globalNames             []string
)

func init() {
    // Load global name data for vectorizer training
    var err error
    loader := data.NewLoader("./")
    globalNames, err = loader.LoadNames("global.json")
    if err != nil {
        log.Fatalf("Failed to load global names: %v", err)
    }

    // Fit vectorizer with global names
    vectorizer.Fit(globalNames)
    log.Println("Server initialized with vectorizer on global names dataset")
}

func main() {
    r := gin.Default()

    // Configure CORS
    r.Use(cors.New(cors.Config{
        AllowOrigins:     []string{"http://localhost:8000"},
        AllowMethods:     []string{"POST", "GET", "OPTIONS"},
        AllowHeaders:     []string{"Origin", "Content-Type"},
        AllowCredentials: true,
    }))

    // Define routes
    r.POST("/", processEntityMatchingRequest)

    // Start server
    log.Println("Starting server on :8080...")
    if err := r.Run(":8080"); err != nil {
        log.Fatalf("Failed to start server: %v", err)
    }
}


func processEntityMatchingRequest(c *gin.Context) {
    // Parse request data
    var item Item
    if err := c.ShouldBindJSON(&item); err != nil {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid input"})
        return
    }

    if item.Query == "" || len(item.Data) == 0 {
        c.JSON(http.StatusBadRequest, gin.H{"error": "Query or data is empty"})
        return
    }

    // Process with homomorphic encryption
    cosineSims, err := computeHECosineSimilarities(item.Query, item.Data)
    if err != nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
        return
    }

    // Convert polynomial vector to float64 slice for JSON response
    // Return response
    c.JSON(http.StatusOK, Response{
        CosineSims: cosineSims,
        QueryEnc:   utils.GenerateTestVector(1024),
    })
}

func computeHECosineSimilarities(query string, store []string) ([]float64, error) {
    // Preprocess data first using existing data cleaning functions
    cleanedQuery := data.CleanCompanyName(query)
    cleanedStore := make([]string, len(store))
    for i, name := range store {
        cleanedStore[i] = data.CleanCompanyName(name)
    }
    
    // Log the cleaning results for debugging
    log.Println("Cleaned query:", cleanedQuery)
    for i, name := range cleanedStore {
        log.Printf("Cleaned store item %d: '%s' -> '%s'", i, store[i], name)
    }
    
    // Transform cleaned data
    queryVector := vectorizer.Transform(cleanedQuery)
    storeVectors := vectorizer.BatchTransform(cleanedStore)

    // Normalize vectors
    utils.NormalizeVector(&queryVector)
    for i := range storeVectors {
        utils.NormalizeVector(&storeVectors[i])
    }

    // Prepare query vector
    queryVectors := make([][]float64, 1)
    queryVectors[0] = queryVector

    // Log expected plaintext similarities for verification:
    log.Println("Computing plaintext similarities for verification:")
    for i, name := range store {
        sim := utils.DotProduct(queryVector, storeVectors[i])
        log.Printf("  %s: %.6f", name, sim)
    }

    // Batch encrypt the query vector
    encryptedQuery := encCtx.BatchEncrypt(queryVectors)
    enc := encryptedQuery[0]
    fmt.Println("Encrypted query vector:", len(enc.Value))

    // Compute cosine similarities using HE
    resultMatrix, err := evalCtx.BatchDotProduct(encryptedQuery, storeVectors)
    if err != nil {
        return nil, err
    }

    // Decrypt the results
    cosineSimilarities := make([]float64, len(store))

    // Each row contains similarities between query and all store vectors
    decryptedBatch := decCtx.BatchDecrypt(resultMatrix[0])
    for j := range store {
        if decryptedBatch[j] == nil {
            cosineSimilarities[j] = -1 // Indicate null values with -1
            continue
        }

        // The cosine similarity is stored in the first element due to InnerSum
        similarity := decryptedBatch[j][0]
        cosineSimilarities[j] = similarity

        // Log for debugging
        log.Printf("HE similarity - %s: %.6f", store[j], similarity)
    }

    return cosineSimilarities, nil
}