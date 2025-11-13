#!/usr/bin/env python3
"""
Automates the setup of a Spark (3.5.7 or 4.0.1) and Hadoop 3.3.6 cluster.

This script is designed to be run on each node (master, worker, or single-node) with a
specified role and Spark version.

Prerequisites (MUST be done manually first):
1.  Java installation:
    - Spark 3.5.7: Requires Java 8, 11, or 17
    - Spark 4.0.1: Requires Java 17 or 21 ONLY (Java 8/11 not supported)
2.  For multi-node: Both nodes must have /etc/hosts configured to resolve each other by name.
3.  For multi-node: The master node MUST have passwordless SSH access to the worker node.
    (e.g., from master: `ssh-copy-id user@worker-host`)

Usage:
1.  Single-node setup:
    uv run setup-spark.py --role single --spark-version 3.5.7
    uv run setup-spark.py --role single --spark-version 4.0.1

2.  Multi-node setup:
    On WORKER: uv run setup-spark.py --role worker --spark-version 3.5.7
    On MASTER: uv run setup-spark.py --role master --spark-version 3.5.7

3.  Clean install (remove existing installation first):
    uv run setup-spark.py --role single --spark-version 4.0.1 --cleanup-existing

4.  Cleanup only (remove installation without reinstalling):
    uv run setup-spark.py --role single --spark-version 3.5.7 --cleanup-only

5.  Full cleanup (remove Spark, Hadoop, and HDFS data):
    uv run setup-spark.py --role single --spark-version 3.5.7 --cleanup-only --full-cleanup
"""

import argparse
import getpass
import os
import re
import subprocess
import shlex
import sys
import tarfile
import urllib.request
import shutil
from pathlib import Path
from datetime import datetime

# --- Configuration ---
# !! Update these hosts to match your /etc/hosts entries !!
MASTER_HOST = "spark-master"
WORKER_HOST = "spark-worker"

# Versions - will be set based on command-line argument
SPARK_VERSION = None  # Will be set in main()
HADOOP_VERSION = "3.3.6"
HADOOP_SPARK_COMBO = "hadoop3"  # Matches the Spark download name

# Paths (User-level installation in home directory)
BASE_DIR = Path.home()
JAVA_HOME_PATH = None  # Will be auto-discovered
HADOOP_HOME = BASE_DIR / f"hadoop-{HADOOP_VERSION}"
SPARK_HOME = None  # Will be set based on SPARK_VERSION in main()

# Java compatibility - will be set based on Spark version
# Spark 3.5.7: Java 8-17
# Spark 4.0.1: Java 17-21
MIN_JAVA_VERSION = None  # Will be set in main()
MAX_JAVA_VERSION = None  # Will be set in main()

# Java version requirements per Spark version
SPARK_JAVA_REQUIREMENTS = {
    "3.5.7": {"min": 8, "max": 17},
    "4.0.1": {"min": 17, "max": 21},
}

# Download URLs - will be built based on version
HADOOP_URLS = None  # Will be set in main()
SPARK_URLS = None  # Will be set in main()

# Data directory for HDFS
HDFS_DATA_DIR = BASE_DIR / "hadoop_data" / "hdfs"

# Log file for recording all actions
LOG_FILE = None  # Will be set in main()

# --- ANSI Color Codes ---
class C:
    OK = '\033[92m'
    WARN = '\033[93m'
    FAIL = '\033[91m'
    HEAD = '\033[95m'
    INFO = '\033[94m'
    END = '\033[0m'

# --- Helper Functions ---

def log_to_file(msg, prefix=""):
    """Writes a message to the log file with timestamp."""
    if LOG_FILE:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, 'a') as f:
            f.write(f"[{timestamp}] {prefix}{msg}\n")

def print_info(msg):
    print(f"{C.INFO}INFO: {msg}{C.END}")
    log_to_file(msg, "INFO: ")

def print_ok(msg):
    print(f"{C.OK}SUCCESS: {msg}{C.END}")
    log_to_file(msg, "SUCCESS: ")

def print_warn(msg):
    print(f"{C.WARN}WARNING: {msg}{C.END}")
    log_to_file(msg, "WARNING: ")

def print_fail(msg):
    print(f"{C.FAIL}ERROR: {msg}{C.END}", file=sys.stderr)
    log_to_file(msg, "ERROR: ")
    sys.exit(1)

def print_header(msg):
    print(f"\n{C.HEAD}--- {msg} ---{C.END}")
    log_to_file(f"\n{'='*60}\n{msg}\n{'='*60}", "")

def run_cmd(command, check=True, shell=False):
    """Runs a shell command."""
    if not shell:
        if isinstance(command, str):
            command = shlex.split(command)
        cmd_str = shlex.join(command)
        print(f"  {C.OK}$> {cmd_str}{C.END}")
    else:
        cmd_str = command
        print(f"  {C.OK}$> {cmd_str}{C.END}")

    log_to_file(f"COMMAND: {cmd_str}", "")

    try:
        subprocess.run(command, check=check, shell=shell)
        log_to_file(f"Command completed successfully", "  ")
    except subprocess.CalledProcessError as e:
        log_to_file(f"Command failed with exit code {e.returncode}", "  ERROR: ")
        print_fail(f"Command failed with exit code {e.returncode}")
    except FileNotFoundError as e:
        log_to_file(f"Command not found: {e.filename}", "  ERROR: ")
        print_fail(f"Command not found: {e.filename}. Is it installed and in your PATH?")

def write_file(path, content):
    """Writes content to a file, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    print(f"  {C.OK}Wrote config: {path}{C.END}")

    # Log the file and its content
    log_to_file(f"WROTE FILE: {path}", "")
    log_to_file(f"--- BEGIN FILE CONTENT ---", "  ")
    for line in content.splitlines():
        log_to_file(line, "  ")
    log_to_file(f"--- END FILE CONTENT ---", "  ")

def verify_tarfile_integrity(filepath):
    """Verifies that a .tar.gz/.tgz file is valid and not corrupted."""
    try:
        with tarfile.open(filepath, "r:gz") as tar:
            # Try to read the member list - this will fail if corrupted
            members = tar.getmembers()
            if len(members) == 0:
                return False
        return True
    except Exception as e:
        log_to_file(f"File integrity check failed: {e}", "WARNING: ")
        return False

def try_download_from_url(url, download_path):
    """Attempts to download a file from a single URL. Returns True on success, False on failure."""
    try:
        print_info(f"Trying: {url}")
        log_to_file(f"Attempting download from: {url}", "DOWNLOAD: ")

        with urllib.request.urlopen(url, timeout=30) as response:
            # Get file size if available
            file_size = response.headers.get('Content-Length')
            if file_size:
                file_size = int(file_size)
                file_size_mb = file_size / (1024 * 1024)
                print_info(f"File size: {file_size_mb:.1f} MB")

            # Download with progress
            downloaded = 0
            block_size = 10 * 1024 * 1024  # 10 MB chunks for better performance
            with open(download_path, 'wb') as out_file:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)

                    # Show progress every 10 MB
                    if file_size and downloaded % (10 * 1024 * 1024) < block_size:
                        progress_pct = (downloaded / file_size) * 100
                        downloaded_mb = downloaded / (1024 * 1024)
                        print(f"  Progress: {downloaded_mb:.1f} MB / {file_size_mb:.1f} MB ({progress_pct:.1f}%)")

        downloaded_mb = downloaded / (1024 * 1024)
        print_ok(f"Downloaded {downloaded_mb:.1f} MB")
        log_to_file(f"Successfully downloaded from: {url}", "SUCCESS: ")
        return True

    except Exception as e:
        print_warn(f"Failed: {e}")
        log_to_file(f"Download failed from {url}: {e}", "WARNING: ")
        # Clean up partial download
        if download_path.exists():
            download_path.unlink()
        return False

def download_and_extract(urls, download_path, extract_dir, force_redownload=False):
    """Downloads and extracts a .tar.gz or .tgz file. Tries multiple mirror URLs in order."""
    # Convert single URL to list for backwards compatibility
    if isinstance(urls, str):
        urls = [urls]

    if extract_dir.exists() and not force_redownload:
        print_info(f"{extract_dir.name} already exists, skipping download.")
        return

    download_path = Path(download_path)

    # Check if file exists and verify integrity
    needs_download = True
    if download_path.exists() and not force_redownload:
        print_info(f"Checking integrity of existing file: {download_path.name}")
        if verify_tarfile_integrity(download_path):
            print_ok("File integrity verified, skipping download.")
            needs_download = False
        else:
            print_warn(f"File appears corrupted: {download_path.name}")
            print_info("Removing corrupt file and re-downloading...")
            log_to_file(f"Removing corrupt file: {download_path}", "")
            download_path.unlink()
    elif force_redownload and download_path.exists():
        print_info(f"Force re-download: removing existing file {download_path.name}")
        log_to_file(f"Force removing existing file: {download_path}", "")
        download_path.unlink()

    # Download if needed
    if needs_download or force_redownload:
        print_info(f"Saving to: {download_path}")
        download_success = False

        for url in urls:
            if try_download_from_url(url, download_path):
                download_success = True
                break

        if not download_success:
            print_fail(f"Failed to download from all mirrors. Tried {len(urls)} URL(s).")

        # Verify the newly downloaded file
        print_info("Verifying downloaded file integrity...")
        if not verify_tarfile_integrity(download_path):
            print_fail(f"Downloaded file is corrupted: {download_path.name}")
        print_ok("Download integrity verified.")

    # Remove existing extracted directory if forcing redownload
    if force_redownload and extract_dir.exists():
        print_info(f"Force re-download: removing existing directory {extract_dir.name}")
        log_to_file(f"Force removing existing directory: {extract_dir}", "")
        shutil.rmtree(extract_dir)

    # Extract the archive
    if not extract_dir.exists():
        print_info(f"Extracting {download_path.name} to {BASE_DIR}...")
        try:
            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(path=BASE_DIR)
            print_ok(f"Extracted to {extract_dir}")
        except Exception as e:
            print_fail(f"Failed to extract {download_path.name}: {e}")
    else:
        print_info(f"Extract directory already exists: {extract_dir.name}")

# --- Java Discovery Functions ---

def get_java_version(java_executable):
    """Returns the major version number of a Java executable, or None if invalid."""
    try:
        result = subprocess.run(
            [str(java_executable), "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Java version is in stderr
        version_output = result.stderr

        # Parse version from output like: java version "17.0.1" or openjdk version "11.0.12"
        # Modern format: java version "17.0.1"
        # Older format: java version "1.8.0_292"
        match = re.search(r'version\s+"([^"]+)"', version_output)
        if match:
            version_str = match.group(1)
            # Handle both "1.8.0" and "17.0.1" formats
            parts = version_str.split('.')
            if parts[0] == '1':
                # Old format: 1.8.0 means Java 8
                return int(parts[1])
            else:
                # New format: 17.0.1 means Java 17
                return int(parts[0])
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, IndexError):
        pass
    return None

def find_java_installations():
    """Searches typical JDK installation locations and returns list of (path, version) tuples."""
    java_installations = []

    # Common JDK installation directories
    search_paths = [
        Path("/usr/lib/jvm"),
        Path("/usr/java"),
        Path("/opt/java"),
        Path("/opt/jdk"),
        Path("/Library/Java/JavaVirtualMachines"),  # macOS
    ]

    # Also check if 'java' is in PATH
    try:
        result = subprocess.run(["which", "java"], capture_output=True, text=True)
        if result.returncode == 0:
            java_path = Path(result.stdout.strip())
            # Resolve symlinks to get actual JAVA_HOME
            java_path = java_path.resolve()
            # JAVA_HOME is typically two levels up from bin/java
            if java_path.parent.name == "bin":
                java_home = java_path.parent.parent
                version = get_java_version(java_path)
                if version:
                    java_installations.append((java_home, version))
    except Exception:
        pass

    # Search common installation directories
    for base_path in search_paths:
        if not base_path.exists():
            continue

        # Look for subdirectories that might be Java installations
        try:
            for item in base_path.iterdir():
                if item.is_dir():
                    # Check if this looks like a Java installation
                    java_executable = item / "bin" / "java"
                    if java_executable.exists():
                        version = get_java_version(java_executable)
                        if version:
                            java_installations.append((item, version))
        except PermissionError:
            continue

    # Remove duplicates (same path)
    seen = set()
    unique_installations = []
    for path, version in java_installations:
        path_str = str(path.resolve())
        if path_str not in seen:
            seen.add(path_str)
            unique_installations.append((path, version))

    return unique_installations

def select_compatible_java():
    """Finds and selects a compatible Java installation for Spark."""
    global JAVA_HOME_PATH

    print_info(f"Searching for compatible Java installation (JDK {MIN_JAVA_VERSION}-{MAX_JAVA_VERSION} for Spark {SPARK_VERSION})...")

    installations = find_java_installations()

    # Log all found Java installations
    log_to_file(f"Found {len(installations)} Java installation(s):", "JAVA DISCOVERY: ")
    for path, ver in installations:
        log_to_file(f"  Java {ver} at {path}", "  ")

    if not installations:
        print_fail(f"No Java installations found. Please install Java JDK {MIN_JAVA_VERSION}-{MAX_JAVA_VERSION}.")

    # Filter for compatible versions (must be within MIN and MAX range)
    compatible = [(path, ver) for path, ver in installations if MIN_JAVA_VERSION <= ver <= MAX_JAVA_VERSION]

    if not compatible:
        print_fail(f"Found Java installations, but none are compatible with Spark {SPARK_VERSION} (requires JDK {MIN_JAVA_VERSION}-{MAX_JAVA_VERSION}):\n" +
                   "\n".join([f"  - {path}: Java {ver}" for path, ver in installations]))

    # Sort by version descending (prefer Java 21 over Java 17, Java 17 over Java 11, etc.)
    compatible.sort(key=lambda x: x[1], reverse=True)

    # Select the best option
    selected_path, selected_version = compatible[0]
    JAVA_HOME_PATH = str(selected_path)

    print_ok(f"Selected Java {selected_version} at: {JAVA_HOME_PATH}")
    log_to_file(f"SELECTED: Java {selected_version} at {JAVA_HOME_PATH}", "JAVA DECISION: ")

    # Show other compatible options if any
    if len(compatible) > 1:
        print_info("Other compatible Java installations found:")
        for path, ver in compatible[1:]:
            print(f"  - {path}: Java {ver}")
            log_to_file(f"  Alternative: Java {ver} at {path}", "  ")

    return JAVA_HOME_PATH

# --- SSH Setup Functions ---

def run_sudo_command(command, password):
    """Runs a command with sudo using the provided password."""
    try:
        # Use sudo -S to read password from stdin
        # Wrap command in sh -c to ensure compound commands (with &&) run entirely under sudo
        full_command = f"sudo -S sh -c {shlex.quote(command)}"
        process = subprocess.Popen(
            full_command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=password + '\n')
        log_to_file(f"SUDO COMMAND: {command}", "")
        if process.returncode == 0:
            log_to_file(f"Command succeeded", "  ")
            return True, stdout
        else:
            log_to_file(f"Command failed: {stderr}", "  ERROR: ")
            return False, stderr
    except Exception as e:
        log_to_file(f"Exception running sudo command: {e}", "ERROR: ")
        return False, str(e)

def detect_package_manager():
    """Detects which package manager is available (apt or yum)."""
    if subprocess.run(["which", "apt-get"], capture_output=True).returncode == 0:
        return "apt"
    elif subprocess.run(["which", "yum"], capture_output=True).returncode == 0:
        return "yum"
    elif subprocess.run(["which", "dnf"], capture_output=True).returncode == 0:
        return "dnf"
    return None

def install_and_start_ssh_server():
    """Attempts to install and start SSH server with user's permission."""
    print_warn("SSH server is not running. Hadoop and Spark require SSH to start services.")
    print_info("")
    response = input("Would you like to install and start SSH server now? (y/n): ").strip().lower()

    if response != 'y':
        print_info("Skipping SSH server installation.")
        print_warn("Please install SSH server manually and run this script again.")
        log_to_file("User declined SSH server installation", "")
        return False

    # Detect package manager
    pkg_mgr = detect_package_manager()
    if not pkg_mgr:
        print_warn("Could not detect package manager (apt-get, yum, or dnf).")
        print_info("Please install openssh-server manually.")
        return False

    print_info(f"Detected package manager: {pkg_mgr}")

    # Prompt for sudo password
    print_info("sudo password required to install SSH server.")
    password = getpass.getpass("Enter your sudo password: ")

    # Test sudo password first
    print_info("Verifying sudo password...")
    success, _ = run_sudo_command("echo 'sudo test'", password)
    if not success:
        print_fail("Incorrect sudo password or insufficient permissions.")
        return False

    print_ok("Sudo password verified.")

    # Install SSH server
    print_info("Installing SSH server...")
    if pkg_mgr == "apt":
        install_cmd = "apt-get update && apt-get install -y openssh-server"
        service_name = "ssh"
    elif pkg_mgr in ["yum", "dnf"]:
        install_cmd = f"{pkg_mgr} install -y openssh-server"
        service_name = "sshd"

    success, output = run_sudo_command(install_cmd, password)
    if not success:
        print_fail(f"Failed to install SSH server: {output}")
        return False

    print_ok("SSH server installed successfully.")

    # Start and enable SSH server
    print_info("Starting SSH server...")
    success, output = run_sudo_command(f"systemctl start {service_name}", password)
    if not success:
        print_warn(f"Failed to start SSH server: {output}")
        return False

    success, output = run_sudo_command(f"systemctl enable {service_name}", password)
    if not success:
        print_warn(f"Failed to enable SSH server: {output}")

    print_ok("SSH server started and enabled.")
    log_to_file("SSH server installed and started successfully", "SUCCESS: ")
    return True

def check_ssh_server():
    """Checks if SSH server is running."""
    try:
        result = subprocess.run(["systemctl", "is-active", "ssh"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip() == "active":
            return True
        # Try sshd as service name (some distros use this)
        result = subprocess.run(["systemctl", "is-active", "sshd"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip() == "active":
            return True
    except Exception:
        pass
    return False

def setup_passwordless_ssh_localhost():
    """Sets up passwordless SSH to localhost if not already configured."""
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(mode=0o700, exist_ok=True)

    id_rsa = ssh_dir / "id_rsa"
    id_rsa_pub = ssh_dir / "id_rsa.pub"
    authorized_keys = ssh_dir / "authorized_keys"

    # Generate SSH key if it doesn't exist
    if not id_rsa.exists():
        print_info("Generating SSH key pair for passwordless authentication...")
        log_to_file("Generating SSH key pair", "SSH SETUP: ")
        run_cmd(f'ssh-keygen -t rsa -P "" -f {id_rsa}', shell=True)
        print_ok("SSH key pair generated.")
    else:
        print_info("SSH key pair already exists.")

    # Add public key to authorized_keys if not already there
    if id_rsa_pub.exists():
        with open(id_rsa_pub, 'r') as f:
            pub_key = f.read().strip()

        # Check if key is already in authorized_keys
        key_exists = False
        if authorized_keys.exists():
            with open(authorized_keys, 'r') as f:
                if pub_key in f.read():
                    key_exists = True

        if not key_exists:
            print_info("Adding public key to authorized_keys...")
            log_to_file("Adding public key to authorized_keys", "SSH SETUP: ")
            with open(authorized_keys, 'a') as f:
                f.write(pub_key + '\n')
            authorized_keys.chmod(0o600)
            print_ok("Public key added to authorized_keys.")
        else:
            print_info("Public key already in authorized_keys.")

    # Test SSH connection to localhost
    print_info("Testing SSH connection to localhost...")
    result = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
         "-o", "ConnectTimeout=5", "localhost", "echo", "SSH_OK"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0 and "SSH_OK" in result.stdout:
        print_ok("Passwordless SSH to localhost is working.")
        log_to_file("Passwordless SSH to localhost verified", "SSH SETUP: ")
        return True
    else:
        print_warn("SSH test failed. Please ensure SSH server is running.")
        log_to_file(f"SSH test failed: {result.stderr}", "WARNING: ")
        return False

def check_etc_hosts():
    """Verifies /etc/hosts has localhost entry."""
    print_info("Checking /etc/hosts for localhost entry...")
    try:
        with open('/etc/hosts', 'r') as f:
            hosts_content = f.read()

        if '127.0.0.1' in hosts_content and 'localhost' in hosts_content:
            print_ok("/etc/hosts contains localhost entry.")
            return True
        else:
            print_warn("/etc/hosts may be missing localhost entry.")
            print_info("Expected entry: 127.0.0.1    localhost")
            return False
    except PermissionError:
        print_warn("Cannot read /etc/hosts (permission denied).")
        return False

# --- Cleanup Functions ---

def stop_spark_services():
    """Stops Spark services if they are running."""
    print_info("Attempting to stop Spark services...")

    # Check if SPARK_HOME exists
    if not SPARK_HOME or not SPARK_HOME.exists():
        print_info("Spark installation not found, skipping Spark service stop.")
        log_to_file("Spark installation not found, skipping service stop", "")
        return

    stop_script = SPARK_HOME / "sbin" / "stop-all.sh"
    if not stop_script.exists():
        print_info("Spark stop script not found, skipping.")
        return

    try:
        print_info("Running stop-all.sh...")
        subprocess.run(
            [str(stop_script)],
            check=False,  # Don't fail if services weren't running
            capture_output=True,
            timeout=30
        )
        print_ok("Spark services stopped (or were not running).")
        log_to_file("Executed stop-all.sh", "CLEANUP: ")
    except subprocess.TimeoutExpired:
        print_warn("Spark stop command timed out.")
    except Exception as e:
        print_warn(f"Error stopping Spark services: {e}")

def stop_hdfs_services():
    """Stops HDFS services if they are running."""
    print_info("Attempting to stop HDFS services...")

    # Check if HADOOP_HOME exists
    if not HADOOP_HOME.exists():
        print_info("Hadoop installation not found, skipping HDFS service stop.")
        log_to_file("Hadoop installation not found, skipping service stop", "")
        return

    stop_script = HADOOP_HOME / "sbin" / "stop-dfs.sh"
    if not stop_script.exists():
        print_info("HDFS stop script not found, skipping.")
        return

    try:
        print_info("Running stop-dfs.sh...")
        subprocess.run(
            [str(stop_script)],
            check=False,  # Don't fail if services weren't running
            capture_output=True,
            timeout=30
        )
        print_ok("HDFS services stopped (or were not running).")
        log_to_file("Executed stop-dfs.sh", "CLEANUP: ")
    except subprocess.TimeoutExpired:
        print_warn("HDFS stop command timed out.")
    except Exception as e:
        print_warn(f"Error stopping HDFS services: {e}")

def remove_spark_installation():
    """Removes the Spark installation directory for the specified version."""
    print_info(f"Removing Spark {SPARK_VERSION} installation...")

    if not SPARK_HOME:
        print_info("SPARK_HOME not set, nothing to remove.")
        return

    if SPARK_HOME.exists():
        print_info(f"Removing directory: {SPARK_HOME}")
        log_to_file(f"Removing Spark installation at {SPARK_HOME}", "CLEANUP: ")
        try:
            shutil.rmtree(SPARK_HOME)
            print_ok(f"Removed {SPARK_HOME}")
        except Exception as e:
            print_warn(f"Error removing Spark directory: {e}")
            log_to_file(f"Error removing Spark directory: {e}", "ERROR: ")
    else:
        print_info(f"Spark installation not found at {SPARK_HOME}")
        log_to_file(f"Spark installation not found at {SPARK_HOME}", "")

def remove_hadoop_installation():
    """Removes the Hadoop installation directory."""
    print_info(f"Removing Hadoop {HADOOP_VERSION} installation...")

    if HADOOP_HOME.exists():
        print_info(f"Removing directory: {HADOOP_HOME}")
        log_to_file(f"Removing Hadoop installation at {HADOOP_HOME}", "CLEANUP: ")
        try:
            shutil.rmtree(HADOOP_HOME)
            print_ok(f"Removed {HADOOP_HOME}")
        except Exception as e:
            print_warn(f"Error removing Hadoop directory: {e}")
            log_to_file(f"Error removing Hadoop directory: {e}", "ERROR: ")
    else:
        print_info(f"Hadoop installation not found at {HADOOP_HOME}")
        log_to_file(f"Hadoop installation not found at {HADOOP_HOME}", "")

def remove_hdfs_data():
    """Removes HDFS data directories."""
    print_info("Removing HDFS data directories...")

    if HDFS_DATA_DIR.exists():
        print_warn(f"This will DELETE ALL DATA in {HDFS_DATA_DIR}")
        response = input("Are you sure you want to remove HDFS data? (yes/no): ").strip().lower()

        if response == 'yes':
            log_to_file(f"User confirmed removal of HDFS data at {HDFS_DATA_DIR}", "CLEANUP: ")
            try:
                shutil.rmtree(HDFS_DATA_DIR)
                print_ok(f"Removed {HDFS_DATA_DIR}")
            except Exception as e:
                print_warn(f"Error removing HDFS data: {e}")
                log_to_file(f"Error removing HDFS data: {e}", "ERROR: ")
        else:
            print_info("Skipped HDFS data removal.")
            log_to_file("User declined HDFS data removal", "")
    else:
        print_info(f"HDFS data directory not found at {HDFS_DATA_DIR}")

def cleanup_downloaded_files():
    """Removes downloaded tarballs."""
    print_info("Removing downloaded tarballs...")

    # Spark tarball
    spark_filename = f"spark-{SPARK_VERSION}-bin-{HADOOP_SPARK_COMBO}.tgz"
    spark_tarball = BASE_DIR / spark_filename
    if spark_tarball.exists():
        print_info(f"Removing {spark_tarball}")
        log_to_file(f"Removing tarball: {spark_tarball}", "CLEANUP: ")
        spark_tarball.unlink()
        print_ok(f"Removed {spark_filename}")

    # Hadoop tarball (only if removing Hadoop too)
    hadoop_filename = f"hadoop-{HADOOP_VERSION}.tar.gz"
    hadoop_tarball = BASE_DIR / hadoop_filename
    if hadoop_tarball.exists():
        print_info(f"Found {hadoop_tarball}")
        response = input(f"Remove Hadoop tarball {hadoop_filename}? (y/n): ").strip().lower()
        if response == 'y':
            log_to_file(f"Removing tarball: {hadoop_tarball}", "CLEANUP: ")
            hadoop_tarball.unlink()
            print_ok(f"Removed {hadoop_filename}")

def phase_cleanup(cleanup_mode='existing'):
    """
    Performs cleanup of existing installations.

    Args:
        cleanup_mode: 'existing' (remove before reinstall) or 'full' (complete removal)
    """
    print_header(f"Cleanup: Removing Spark {SPARK_VERSION} Installation")

    # Always stop services first
    stop_spark_services()
    stop_hdfs_services()

    # Remove Spark installation
    remove_spark_installation()

    # Remove downloaded Spark tarball
    spark_filename = f"spark-{SPARK_VERSION}-bin-{HADOOP_SPARK_COMBO}.tgz"
    spark_tarball = BASE_DIR / spark_filename
    if spark_tarball.exists():
        print_info(f"Removing downloaded tarball: {spark_filename}")
        log_to_file(f"Removing tarball: {spark_tarball}", "CLEANUP: ")
        spark_tarball.unlink()
        print_ok(f"Removed {spark_filename}")

    # For full cleanup mode, offer to remove Hadoop and HDFS data
    if cleanup_mode == 'full':
        print_info("\nFull cleanup mode: You can also remove Hadoop and HDFS data.")

        response = input("Remove Hadoop installation? (y/n): ").strip().lower()
        if response == 'y':
            remove_hadoop_installation()

            # Offer to remove Hadoop tarball
            hadoop_filename = f"hadoop-{HADOOP_VERSION}.tar.gz"
            hadoop_tarball = BASE_DIR / hadoop_filename
            if hadoop_tarball.exists():
                response = input(f"Remove Hadoop tarball {hadoop_filename}? (y/n): ").strip().lower()
                if response == 'y':
                    log_to_file(f"Removing tarball: {hadoop_tarball}", "CLEANUP: ")
                    hadoop_tarball.unlink()
                    print_ok(f"Removed {hadoop_filename}")

        # Offer to remove HDFS data
        if HDFS_DATA_DIR.exists():
            remove_hdfs_data()

    print_ok(f"Cleanup completed for Spark {SPARK_VERSION}")

# --- Phase Functions ---

def phase_check_prereqs():
    print_header("Phase 1: Checking Prerequisites")

    # Discover and select compatible Java
    select_compatible_java()

    # Check for SSH client
    if subprocess.run(["which", "ssh"], capture_output=True).returncode != 0:
        print_fail("`ssh` command not found. Please install OpenSSH client.")
    print_ok("`ssh` client found.")

    # Check for SSH server
    print_info("Checking SSH server status...")
    ssh_server_running = check_ssh_server()

    if not ssh_server_running:
        # Offer to install and start SSH server
        if not install_and_start_ssh_server():
            print_warn("Continuing without SSH server. Services may not start correctly.")
            log_to_file("SSH server not running - continuing anyway", "WARNING: ")
        else:
            # Verify SSH server is now running
            ssh_server_running = check_ssh_server()
            if not ssh_server_running:
                print_warn("SSH server installation completed but server is not running.")
    else:
        print_ok("SSH server is running.")

    # Check /etc/hosts
    check_etc_hosts()

    # Setup passwordless SSH to localhost
    print_info("Setting up passwordless SSH to localhost...")
    if not setup_passwordless_ssh_localhost():
        if not check_ssh_server():
            print_warn("Skipping SSH setup because SSH server is not running.")
        else:
            print_warn("Could not verify passwordless SSH. Services may not start correctly.")

    # Check for wget/curl (for download)
    if subprocess.run(["which", "curl"], capture_output=True).returncode != 0:
        print_warn("`curl` not found. Using Python's urllib.")

    print_ok("Prerequisite check completed.")

def phase_download_and_extract(force_redownload=False):
    print_header("Phase 2: Downloading and Extracting")
    if force_redownload:
        print_warn("Force re-download enabled: will download all files fresh")
        log_to_file("Force re-download flag is set", "")

    # Use the first URL to extract the filename (all mirrors have same filename)
    hadoop_filename = HADOOP_URLS[0].split('/')[-1]
    spark_filename = SPARK_URLS[0].split('/')[-1]

    download_and_extract(HADOOP_URLS, BASE_DIR / hadoop_filename, HADOOP_HOME, force_redownload)
    download_and_extract(SPARK_URLS, BASE_DIR / spark_filename, SPARK_HOME, force_redownload)

def phase_update_bashrc():
    print_header("Phase 3: Updating .bashrc")
    bashrc_path = BASE_DIR / ".bashrc"

    bashrc_content = f"""
# --- Spark/Hadoop Cluster Config ---
export JAVA_HOME={JAVA_HOME_PATH}
export HADOOP_HOME={HADOOP_HOME}
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export SPARK_HOME={SPARK_HOME}
export PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin
# --- End Spark/Hadoop Config ---
"""

    # Set environment variables for current session
    os.environ['JAVA_HOME'] = JAVA_HOME_PATH
    os.environ['HADOOP_HOME'] = str(HADOOP_HOME)
    os.environ['HADOOP_CONF_DIR'] = str(HADOOP_HOME / "etc" / "hadoop")
    os.environ['SPARK_HOME'] = str(SPARK_HOME)
    os.environ['PATH'] = f"{os.environ['PATH']}:{JAVA_HOME_PATH}/bin:{HADOOP_HOME}/bin:{HADOOP_HOME}/sbin:{SPARK_HOME}/bin"
    print_ok("Environment variables set for current session.")

    # Log environment variables
    log_to_file("ENVIRONMENT VARIABLES SET:", "")
    log_to_file(f"  JAVA_HOME={JAVA_HOME_PATH}", "  ")
    log_to_file(f"  HADOOP_HOME={HADOOP_HOME}", "  ")
    log_to_file(f"  HADOOP_CONF_DIR={HADOOP_HOME / 'etc' / 'hadoop'}", "  ")
    log_to_file(f"  SPARK_HOME={SPARK_HOME}", "  ")
    log_to_file(f"  PATH appended with: {JAVA_HOME_PATH}/bin:{HADOOP_HOME}/bin:{HADOOP_HOME}/sbin:{SPARK_HOME}/bin", "  ")

    # Add to .bashrc if not already present
    if bashrc_path.exists():
        with open(bashrc_path, 'r') as f:
            if "Spark/Hadoop Cluster Config" in f.read():
                print_info(".bashrc already configured.")
                return

    with open(bashrc_path, 'a') as f:
        f.write(bashrc_content)
    print_ok(f"Appended config to {bashrc_path}")
    print_warn("Please run 'source ~/.bashrc' or re-login for changes to take effect in new shells.")

def phase_generate_hadoop_configs(role):
    print_header("Phase 4: Generating Hadoop Configs")
    conf_dir = HADOOP_HOME / "etc" / "hadoop"

    # Set hostnames based on role
    if role == 'single':
        master_hostname = "localhost"
        worker_hostname = "localhost"
    else:
        master_hostname = MASTER_HOST
        worker_hostname = WORKER_HOST

    # hadoop-env.sh
    write_file(conf_dir / "hadoop-env.sh", f"export JAVA_HOME={JAVA_HOME_PATH}\n")

    # core-site.xml
    core_site_xml = f"""<configuration>
    <property>
        <name>fs.defaultFS</name>
        <value>hdfs://{master_hostname}:9000</value>
    </property>
</configuration>
"""
    write_file(conf_dir / "core-site.xml", core_site_xml)

    # hdfs-site.xml
    # Create the data dirs first
    (HDFS_DATA_DIR / "namenode").mkdir(parents=True, exist_ok=True)
    (HDFS_DATA_DIR / "datanode").mkdir(parents=True, exist_ok=True)
    
    hdfs_site_xml = f"""<configuration>
    <property>
        <name>dfs.replication</name>
        <value>1</value>
    </property>
    <property>
        <name>dfs.namenode.name.dir</name>
        <value>file://{HDFS_DATA_DIR / "namenode"}</value>
    </property>
    <property>
        <name>dfs.datanode.data.dir</name>
        <value>file://{HDFS_DATA_DIR / "datanode"}</value>
    </property>
</configuration>
"""
    write_file(conf_dir / "hdfs-site.xml", hdfs_site_xml)

    # workers
    write_file(conf_dir / "workers", f"{worker_hostname}\n")

def phase_generate_spark_configs(role):
    print_header("Phase 5: Generating Spark Configs")
    conf_dir = SPARK_HOME / "conf"

    # Set hostnames based on role
    if role == 'single':
        master_hostname = "localhost"
        worker_hostname = "localhost"
    else:
        master_hostname = MASTER_HOST
        worker_hostname = WORKER_HOST

    # spark-env.sh
    spark_env_sh = f"""#!/usr/bin/env bash
export JAVA_HOME={JAVA_HOME_PATH}
export HADOOP_CONF_DIR={HADOOP_HOME}/etc/hadoop
export SPARK_MASTER_HOST={master_hostname}
"""
    write_file(conf_dir / "spark-env.sh", spark_env_sh)

    # workers
    write_file(conf_dir / "workers", f"{worker_hostname}\n")

def phase_master_tasks():
    print_header("Phase 6 (Master): Copying Configs to Worker")
    
    # Copy Hadoop config
    hadoop_conf_local = HADOOP_HOME / "etc" / "hadoop"
    hadoop_conf_remote = HADOOP_HOME / "etc"
    cmd = ["scp", "-r", str(hadoop_conf_local), f"{WORKER_HOST}:{hadoop_conf_remote}"]
    run_cmd(cmd)

    # Copy Spark config
    spark_conf_local = SPARK_HOME / "conf"
    spark_conf_remote = SPARK_HOME
    cmd = ["scp", "-r", str(spark_conf_local), f"{WORKER_HOST}:{spark_conf_remote}"]
    run_cmd(cmd)
    
    print_ok("Configs copied to worker.")

    print_header("Phase 7 (Master): Formatting HDFS and Starting Services")
    
    # Format HDFS NameNode (ONLY IF NOT ALREADY FORMATTED)
    format_check_file = HDFS_DATA_DIR / "namenode" / "current" / "VERSION"
    if not format_check_file.exists():
        print_info("Formatting HDFS NameNode (one-time step)...")
        run_cmd(f"{HADOOP_HOME / 'bin' / 'hdfs'} namenode -format -nonInteractive")
        print_ok("HDFS NameNode formatted.")
    else:
        print_info("HDFS NameNode appears to be formatted. Skipping.")

    # Start HDFS
    print_info("Starting HDFS services (start-dfs.sh)...")
    run_cmd(f"{HADOOP_HOME / 'sbin' / 'start-dfs.sh'}", shell=True)

    # Start Spark
    print_info("Starting Spark services (start-all.sh)...")
    run_cmd(f"{SPARK_HOME / 'sbin' / 'start-all.sh'}", shell=True)

    print_header("Cluster Setup Complete!")
    print_ok(f"  HDFS NameNode UI: http://{MASTER_HOST}:9870")
    print_ok(f"  Spark Master UI:    http://{MASTER_HOST}:8080")
    print_warn("\nRemember to run 'source ~/.bashrc' on all nodes if this is your first time.")

def phase_single_node_tasks():
    print_header("Phase 6 (Single-Node): Formatting HDFS and Starting Services")
    
    # Format HDFS NameNode (ONLY IF NOT ALREADY FORMATTED)
    format_check_file = HDFS_DATA_DIR / "namenode" / "current" / "VERSION"
    if not format_check_file.exists():
        print_info("Formatting HDFS NameNode (one-time step)...")
        run_cmd(f"{HADOOP_HOME / 'bin' / 'hdfs'} namenode -format -nonInteractive")
        print_ok("HDFS NameNode formatted.")
    else:
        print_info("HDFS NameNode appears to be formatted. Skipping.")

    # Start HDFS
    print_info("Starting HDFS services (start-dfs.sh)...")
    run_cmd(f"{HADOOP_HOME / 'sbin' / 'start-dfs.sh'}", shell=True)

    # Start Spark
    print_info("Starting Spark services (start-all.sh)...")
    run_cmd(f"{SPARK_HOME / 'sbin' / 'start-all.sh'}", shell=True)

    print_header("Single-Node Cluster Setup Complete!")
    print_ok(f"  HDFS NameNode UI: http://localhost:9870")
    print_ok(f"  Spark Master UI:    http://localhost:8080")
    print_warn("\nRemember to run 'source ~/.bashrc' if this is your first time.")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Spark/Hadoop Cluster Setup Script - Supports Spark 3.5.7 and 4.0.1"
    )
    parser.add_argument(
        '--role',
        choices=['master', 'worker', 'single'],
        required=True,
        help="Specify the role of this node ('master', 'worker', or 'single')."
    )
    parser.add_argument(
        '--spark-version',
        choices=['3.5.7', '4.0.1'],
        required=True,
        help="Specify Spark version to install (3.5.7 or 4.0.1). Java requirements: 3.5.7 needs Java 8-17, 4.0.1 needs Java 17-21."
    )
    parser.add_argument(
        '--redownload-all',
        action='store_true',
        help="Force re-download all files even if they already exist."
    )
    parser.add_argument(
        '--cleanup-existing',
        action='store_true',
        help="Stop services and remove existing Spark installation before installing. Keeps Hadoop and HDFS data intact."
    )
    parser.add_argument(
        '--cleanup-only',
        action='store_true',
        help="Only perform cleanup (stop services, remove installation) and exit. Does not install. Use with --full-cleanup to also remove Hadoop."
    )
    parser.add_argument(
        '--full-cleanup',
        action='store_true',
        help="When used with --cleanup-only, offers to remove Hadoop installation and HDFS data (interactive prompts)."
    )
    args = parser.parse_args()
    role = args.role
    spark_version = args.spark_version
    force_redownload = args.redownload_all
    cleanup_existing = args.cleanup_existing
    cleanup_only = args.cleanup_only
    full_cleanup = args.full_cleanup

    # Validate argument combinations
    if full_cleanup and not cleanup_only:
        print_fail("--full-cleanup can only be used with --cleanup-only")
    if cleanup_only and force_redownload:
        print_warn("--redownload-all is ignored when using --cleanup-only")

    # Set global variables based on Spark version
    global SPARK_VERSION, MIN_JAVA_VERSION, MAX_JAVA_VERSION, SPARK_HOME, HADOOP_URLS, SPARK_URLS

    SPARK_VERSION = spark_version

    # Set Java version requirements
    if SPARK_VERSION not in SPARK_JAVA_REQUIREMENTS:
        print_fail(f"Unsupported Spark version: {SPARK_VERSION}")

    MIN_JAVA_VERSION = SPARK_JAVA_REQUIREMENTS[SPARK_VERSION]["min"]
    MAX_JAVA_VERSION = SPARK_JAVA_REQUIREMENTS[SPARK_VERSION]["max"]

    # Set SPARK_HOME path
    SPARK_HOME = BASE_DIR / f"spark-{SPARK_VERSION}-bin-{HADOOP_SPARK_COMBO}"

    # Build download URLs
    HADOOP_URLS = [
        f"https://dlcdn.apache.org/hadoop/common/hadoop-{HADOOP_VERSION}/hadoop-{HADOOP_VERSION}.tar.gz",
        f"https://downloads.apache.org/hadoop/common/hadoop-{HADOOP_VERSION}/hadoop-{HADOOP_VERSION}.tar.gz",
        f"https://archive.apache.org/dist/hadoop/common/hadoop-{HADOOP_VERSION}/hadoop-{HADOOP_VERSION}.tar.gz",
    ]

    SPARK_URLS = [
        f"https://dlcdn.apache.org/spark/spark-{SPARK_VERSION}/spark-{SPARK_VERSION}-bin-{HADOOP_SPARK_COMBO}.tgz",
        f"https://downloads.apache.org/spark/spark-{SPARK_VERSION}/spark-{SPARK_VERSION}-bin-{HADOOP_SPARK_COMBO}.tgz",
        f"https://archive.apache.org/dist/spark/spark-{SPARK_VERSION}/spark-{SPARK_VERSION}-bin-{HADOOP_SPARK_COMBO}.tgz",
    ]

    # Initialize log file
    global LOG_FILE
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = BASE_DIR / f"spark-hadoop-setup_{role}_{SPARK_VERSION}_{timestamp}.log"

    # Create log file with header
    with open(LOG_FILE, 'w') as f:
        f.write(f"="*80 + "\n")
        f.write(f"Spark/Hadoop Cluster Setup Log\n")
        f.write(f"Role: {role.upper()}\n")
        f.write(f"Spark Version: {SPARK_VERSION}\n")
        f.write(f"Hadoop Version: {HADOOP_VERSION}\n")
        f.write(f"Java Requirements: JDK {MIN_JAVA_VERSION}-{MAX_JAVA_VERSION}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"="*80 + "\n\n")

    print_header(f"Starting Setup for: {role.upper()} NODE - Spark {SPARK_VERSION}")
    print_info(f"Logging to: {LOG_FILE}")

    # Handle cleanup-only mode
    if cleanup_only:
        cleanup_mode = 'full' if full_cleanup else 'existing'
        phase_cleanup(cleanup_mode)

        # Final log summary
        log_to_file("\n" + "="*80, "")
        log_to_file("Cleanup completed successfully", "SUMMARY: ")
        log_to_file(f"All actions have been logged to: {LOG_FILE}", "")
        log_to_file("="*80, "")

        print_info(f"\nDetailed log saved to: {LOG_FILE}")
        print_ok("Cleanup complete. Exiting without installation.")
        return

    # Handle cleanup before installation
    if cleanup_existing:
        phase_cleanup('existing')
        print_info("\nProceeding with fresh installation...\n")

    # Tasks for ALL nodes
    phase_check_prereqs()
    phase_download_and_extract(force_redownload)
    phase_update_bashrc()
    phase_generate_hadoop_configs(role)
    phase_generate_spark_configs(role)

    if role == 'master':
        # Master-only tasks
        phase_master_tasks()
    elif role == 'worker':
        # Worker-only tasks
        print_header("Worker Setup Complete")
        print_info(f"This node ({WORKER_HOST}) is configured.")
        print_info("Please run this script with '--role master' on your master node.")
    elif role == 'single':
        # Single-node tasks
        phase_single_node_tasks()

    # Final log summary
    log_to_file("\n" + "="*80, "")
    log_to_file("Setup completed successfully", "SUMMARY: ")
    log_to_file(f"All commands and configurations have been logged to: {LOG_FILE}", "")
    log_to_file("="*80, "")

    print_info(f"\nDetailed log saved to: {LOG_FILE}")
    print_info("This log contains all commands executed and configuration files created.")

if __name__ == "__main__":
    main()
