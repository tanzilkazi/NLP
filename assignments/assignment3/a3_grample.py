import os

BREAK_COUNT = 500

def conll2string(file):
    text = ""
    count = 0
    with open(INPUT_FILE) as file:
        for line in file:
            if len(line) == 1: # newline
                text = text + " "+ '\n'
            else:
                text = text + " " + line.split(sep=" ")[0]
            count = count + 1
            if count == BREAK_COUNT:
                break
    return text

if __name__ == "__main__":

    INPUT_FILE = r".\conll03\eng.testb"
    OUTPUT_FILE = r".\glample\grample_input.txt"
    text = conll2string(INPUT_FILE)

    with open(OUTPUT_FILE,"w") as f:
        f.write(text)
    print("Output written to file "+ text)
