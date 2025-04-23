import subprocess

# Optional: use timestamped filenames to avoid overwriting
filename = f"test.jpg"

# Run libcamera-jpeg command
cmd = ["libcamera-jpeg", "-o", filename]

try:
    subprocess.run(cmd, check=True)
    print(f"Image captured and saved as {filename}")
except subprocess.CalledProcessError as e:
    print(f"Failed to capture image: {e}")
