package test

import "os"

//WriteToFile writes an array of lines to a file
func WriteToFile(file *os.File, lines []string) error {

	for _, line := range lines {
		if _, err := file.WriteString(line + "\n"); err != nil {
			return err
		}
	}

	return nil
}
