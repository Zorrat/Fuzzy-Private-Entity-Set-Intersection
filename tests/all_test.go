package tests

import (
	"log"
	"os"
	"testing"

	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/compression"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/data"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/utils"
)

func TestAll(t *testing.T) {
	path, _ := os.Getwd()
	loader := data.NewLoader(path)
	names1, err := loader.LoadNames("names1.txt")
	if err != nil {
		log.Fatal(err)
	}
	names2, err := loader.LoadNames("names2.txt")
	if err != nil {
		log.Fatal(err)
	}
	global, err := loader.LoadNames("global.json")
	if err != nil {
		log.Fatal(err)
	}
	log.Print("Loaded names, Vectorizing...")
	vectorizer := data.NewTfidfVectorizer(2, 1)
	vectorizer.Fit(global)

	tfidf1 := vectorizer.BatchTransform(names1)
	tfidf2 := vectorizer.BatchTransform(names2)

	tfidf1_prepped := compression.Prepare(tfidf1)
	tfidf2_prepped := compression.Prepare(tfidf2)

	utils.BatchApply(tfidf1_prepped, compression.FFT)
	utils.BatchApply(tfidf2_prepped, compression.FFT)

	utils.BatchApply(tfidf1_prepped, compression.HighPassFilter, 128)
	utils.BatchApply(tfidf2_prepped, compression.HighPassFilter, 128)

	tfidf1_encoded := compression.ToFloat64(tfidf1_prepped)
	tfidf2_encoded := compression.ToFloat64(tfidf2_prepped)

	// Test the cosine distance
	distances := make([][]float64, len(tfidf1_encoded))
	for i, vec1 := range tfidf1_encoded {
		distances[i] = make([]float64, len(tfidf2_encoded))
		for j, vec2 := range tfidf2_encoded {
			distances[i][j] = utils.CosineDistance(vec1, vec2)
		}
	}
	log.Print("True Negatives")
	// display names if any of the diagnol elements are less than 0.75
	for i := 0; i < len(distances); i++ {
		if distances[i][i] < 0.75 {
			log.Printf("Found a match: %s <-> %s", names1[i], names2[i])
		}
	}
	log.Print("False Positives")
	// display names if any of the off-diagonal elements are greater than 0.75
	for i := 0; i < len(distances); i++ {
		for j := 0; j < len(distances[i]); j++ {
			if i != j && distances[i][j] > 0.75 {
				log.Printf("Found a match: %s <-> %s", names1[i], names2[j])
			}
		}
	}
}
