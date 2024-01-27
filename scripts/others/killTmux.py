import argparse
import libtmux


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("substr", help="kill all tmux sessions with this substring in their name")
    args = parser.parse_args()

    server = libtmux.Server()
    for session in server.sessions:
        if args.substr in session.name:
            session.kill_session()
    
