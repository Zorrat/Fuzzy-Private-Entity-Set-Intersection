package tests

import (
	"log"
	"os"
	"testing"

	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/clustering"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/compression"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/data"
	"github.com/Zorrat/Fuzzy-Private-Entity-Set-Intersection.git/utils"
	"github.com/stretchr/testify/assert"
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

	assert.Equal(t, len(tfidf1_encoded), 100, "Sorted vectors should have the same count as input")
	assert.Equal(t, len(tfidf2_encoded), 100, "Sorted vectors should have the same count as input")

	
}