import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()
    print('hello')
    print(sys.argv[0])
    print(sys.argv[1])
    print(len(sys.argv))
