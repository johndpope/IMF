import os
import pty
import subprocess
import sys
import time
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

def run_in_new_terminal(command):
    master, slave = pty.openpty()
    terminal_process = subprocess.Popen(
        ['gnome-terminal', '--window', '--maximize', '--', 'zsh', '-c', f'{command}; exec bash'],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True
    )
    os.close(slave)
    return master, terminal_process

def main():
    # Command to run in the new terminal
    command = 'python3 -c "import time, sys; [sys.stdout.write(f\'\\rProcessing: {i}%\') or sys.stdout.flush() or time.sleep(0.1) for i in range(101)]"'

    # Start a new terminal window
    master, _ = run_in_new_terminal(command)

    # Create a Rich console that writes to the pseudo-terminal
    console = Console(file=os.fdopen(master, 'w'))

    # Your Rich layout and updating logic here
    with Live(console=console, refresh_per_second=10) as live:
        for i in range(101):
            live.update(Panel(f"Progress: {i}%"))
            time.sleep(0.1)

if __name__ == "__main__":
    main()