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

	names1, err := loader.LoadNames("senderList.json")
	if err != nil {
		log.Fatal(err)
	}
	names2, err := loader.LoadNames("receiverList.json")
	if err != nil {
		log.Fatal(err)
	}
	totalNames, err := loader.LoadNames("global.json")
	if err != nil {
		log.Fatal(err)
	}

	log.Print("Loaded names, Vectorizing...")
	vectorizer := data.NewTfidfVectorizer(2, 1)
	vectorizer.Fit(totalNames)

	tfidf1 := vectorizer.BatchTransform(names1)
	tfidf2 := vectorizer.BatchTransform(names2)
	log.Printf("Original sizes - tfidf1: %dx%d, tfidf2: %dx%d", len(tfidf1), len(tfidf1[0]), len(tfidf2), len(tfidf2[0]))

	d1 := utils.CosineDistanceAll(tfidf1, tfidf2)

	tfidfPrepped1 := compression.Prepare(tfidf1)
	tfidfPrepped2 := compression.Prepare(tfidf2)
	log.Printf("After Prepare - tfidf1: %dx%d, tfidf2: %dx%d", len(tfidfPrepped1), len(tfidfPrepped1[0]), len(tfidfPrepped2), len(tfidfPrepped2[0]))

	utils.BatchApply(tfidfPrepped1, compression.FFT)
	utils.BatchApply(tfidfPrepped2, compression.FFT)
	log.Printf("After FFT - tfidf1: %dx%d, tfidf2: %dx%d", len(tfidfPrepped1), len(tfidfPrepped1[0]), len(tfidfPrepped2), len(tfidfPrepped2[0]))

	// Apply HighPassFilter
	for i := range tfidfPrepped1 {
		tfidfPrepped1[i] = compression.HighPassFilter(tfidfPrepped1[i], 128)
	}
	for i := range tfidfPrepped2 {
		tfidfPrepped2[i] = compression.HighPassFilter(tfidfPrepped2[i], 128)
	}

	log.Printf("After HighPassFilter - tfidf1: %dx%d, tfidf2: %dx%d", len(tfidfPrepped1), len(tfidfPrepped1[0]), len(tfidfPrepped2), len(tfidfPrepped2[0]))

	tfidfCompressed1 := compression.ToFloat64(tfidfPrepped1)
	tfidfCompressed2 := compression.ToFloat64(tfidfPrepped2)
	log.Printf("After ToFloat64 - tfidf1: %dx%d, tfidf2: %dx%d", len(tfidfCompressed1), len(tfidfCompressed1[0]), len(tfidfCompressed2), len(tfidfCompressed2[0]))

	d2 := utils.CosineDistanceAll(tfidfCompressed1, tfidfCompressed2)
	d1D2Mae := utils.MeanAverageError(d1, d2)

	log.Printf("MAE: %f", d1D2Mae)
}
