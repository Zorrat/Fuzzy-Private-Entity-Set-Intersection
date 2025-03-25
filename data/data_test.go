package data

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoader(t *testing.T) {
	loader := NewDataLoader()
	data, err := loader.LoadDataFromTxt("data.txt")
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, 2, len(data))
	assert.Equal(t, "Hello", data[0])
	assert.Equal(t, "World!", data[1])
}
