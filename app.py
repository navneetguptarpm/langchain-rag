import sys
import subprocess
import os

# Get the directory where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

commands = {
    "ingest":   (os.path.join(BASE_DIR, "rag/_ingestion_pipeline.py"), "Running ingestion pipeline to build knowledge base..."),
    "basic":    (os.path.join(BASE_DIR, "rag/_retrieval_pipeline.py"),          "Starting basic RAG chatbot..."),
    "history":  (os.path.join(BASE_DIR, "rag/_history_aware_retrieval_pipeline.py"),  "Starting history-aware RAG chatbot..."),
}

def print_help():
    print("\nUsage: python app.py <command>")
    print("\nAvailable commands:")
    for cmd, (_, description) in commands.items():
        print(f"  {cmd:<10} {description}")
    print()

def main():
    if len(sys.argv) < 2:
        print("No command provided.")
        print_help()
        return

    command = sys.argv[1].lower()

    if command in ("--help", "-h", "help"):
        print_help()
        return

    if command not in commands:
        print(f"Unknown command: '{command}'")
        print_help()
        return

    script, message = commands[command]
    print(f"\n{message}\n")

    try:
        subprocess.run(["python3", script], check=True)
    except KeyboardInterrupt:
        print("\n\nProcess interrupted. Goodbye! 👋")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Script '{script}' exited with error code {e.returncode}.")
    except FileNotFoundError:
        print(f"\nError: Script '{script}' not found. Make sure the file exists.")

if __name__ == "__main__":
    main()