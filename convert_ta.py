from collections import defaultdict

def convert():
    output = ""
    with open("2.10.33_ta.txt") as f:
        for line in f:
            numbers = ''.join(x for x in line if x not in "\"[] ")
            for i in numbers:
                if str(i) in "0123456789":
                    output += str(i)
                    # print ("integer: %i ", i)
                elif i == ",":
                    # print("comma: %s", i)
                    output += "\n"
                else:
                    print ("Unexpected character: %s", i)
    with open("2.10.33_ta2.txt", "w") as f:
        f.write(output)

def enumerate():
    output = defaultdict(lambda:0)

    with open("2.10.33_ta2.txt") as f:
        for line in f:
            output[line] = output[line] + 1
    sortedoutput = sorted(output.items(), key=lambda kv:kv[1], reverse=True)
    with open("2.10.33_ta3.txt", "w") as f:
        f.write(str(sortedoutput))

def main():
    # convert()
    # enumerate()
    
if __name__ == '__main__':
    main()