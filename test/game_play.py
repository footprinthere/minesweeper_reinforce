from game import GameBoard, OpenResult


SIZE = (10, 10)
N_MINES = 10

PRINT_HIDDEN = False


def main():
    gameboard = GameBoard(*SIZE, N_MINES)
    result = OpenResult.OK

    while result == OpenResult.OK:
        gameboard.print(print_hidden=PRINT_HIDDEN)
        x, y = get_position()
        result = gameboard.open(x, y)

    print("*** GAME ENDED: ", result.name)
    gameboard.print(print_hidden=True)


def get_position() -> tuple[int, int]:
    while True:
        inp = input("Enter position to open (x y) >> ")
        tokens = inp.split()

        if len(tokens) != 2:
            print(f"Expected 2 tokens, got {len(tokens)}")
            continue

        try:
            return int(tokens[0]), int(tokens[1])
        except ValueError:
            print(f"Expected integers, got {tokens}")
            continue


if __name__ == "__main__":
    main()
