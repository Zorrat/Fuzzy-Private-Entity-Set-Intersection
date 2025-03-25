package data

import (
	"bufio"
	"fmt"
	"os"
)

type DataLoader struct {
}

func (d *DataLoader) LoadDataFromTxt(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	// Preallocate memory for 10_000 lines
	lines := make([]string, 0, 10_000)

	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	return lines, nil
}

func NewDataLoader() *DataLoader {
	return &DataLoader{}
}
