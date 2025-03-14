import os, pty, select, multiprocessing, threading


def mkpty():
    main1, out = pty.openpty()
    output1 = os.ttyname(out)

    main2, out = pty.openpty()
    output2 = os.ttyname(out)

    print("\ndevice names: ", output1, output2)
    return main1, main2


def run():
    main1, main2 = mkpty()

    def forward(main1, main2):
        while True:
            data = os.read(main1, 128)
            print("read %d data." % len(data))
            os.write(main2, data)

    p1 = threading.Thread(target=forward, args=(main1, main2))
    p1.start()

    def backward(main1, main2):
        while True:
            data = os.read(main2, 128)
            print("read %d data." % len(data))
            os.write(main1, data)

    p2 = threading.Thread(target=backward, args=(main1, main2))
    p2.start()


if __name__ == "__main__":
    run()

    exit()
    main1, main2 = mkpty()
    while True:
        rl, wl, el = select.select([main1, main2], [], [], 1)
        for output in rl:
            data = os.read(output, 128)
            print("read %d data." % len(data))
            if output == main1:
                os.write(main2, data)
            else:
                os.write(main1, data)
