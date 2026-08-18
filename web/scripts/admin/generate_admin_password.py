#!/usr/bin/env python3
"""Generate a Werkzeug password hash for the admin dashboard."""

import argparse
import getpass

from werkzeug.security import generate_password_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--password",
        help="Password to hash; omit this option to enter it without terminal echo",
    )
    args = parser.parse_args()

    password = args.password or getpass.getpass("Admin password: ")
    if not password:
        raise SystemExit("Password must not be empty.")

    password_hash = generate_password_hash(password)
    print("Generated ADMIN_PASSWORD_HASH:")
    print(password_hash)


if __name__ == "__main__":
    main()
