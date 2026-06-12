import sys
import zipfile


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print("usage: unzip -Z1 <zip> | unzip -p <zip> <entry>", file=sys.stderr)
        return 2

    mode = args[0]
    archive = args[1]
    try:
        with zipfile.ZipFile(archive) as zf:
            if mode == "-Z1":
                sys.stdout.write("\n".join(zf.namelist()))
                if zf.namelist():
                    sys.stdout.write("\n")
                return 0
            if mode == "-p" and len(args) >= 3:
                sys.stdout.buffer.write(zf.read(args[2]))
                return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("unsupported unzip arguments: " + " ".join(args), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
