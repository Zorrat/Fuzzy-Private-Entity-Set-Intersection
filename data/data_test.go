package fpsi

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)


func TestJsonLoader(t *testing.T) {
	names,err := load_names("names_train.json")
	if err != nil {
		fmt.Println(err)
	}
	
	fmt.Printf("Loaded names count: %d\n", len(names))
	assert.Equal(t, 8000, len(names), "Expected 1000 names")
}
func TestTxtLoader(t *testing.T) {
	names,err := load_names("names_train.txt")
	if err != nil {
		fmt.Println(err)
	}
	fmt.Printf("Loaded names count: %d\n", len(names))
	assert.Equal(t, 6, len(names), "Expected 1000 names")
}

func TestVectorizer(t *testing.T) {
	names,err := load_names("names_train.json")
	if err != nil {
		fmt.Println(err)
	}
	
	v := NewTfidfVectorizer(2, 1)
	tfidfMatrix := v.FitTransform(names)
	fmt.Println(len(tfidfMatrix))
	assert.Equal(t, len(names), len(tfidfMatrix), "Expected 8000 names")

}