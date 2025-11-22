#!/usr/bin/env bash
set -euo pipefail

# download_checkpoint.sh
# Helper to download and extract a model checkpoint into a target folder.
# Supports: direct URL downloads, Google Drive via gdown, Hugging Face repo via git (with lfs), and AWS S3.

usage() {
  cat <<EOF
Usage: $0 [--url URL] [--gdrive ID] [--hf REPO_URL] [--s3 S3_PATH] --outdir DIR

Options:
  --url URL         Direct download URL (http/https). Archive (.zip, .tar.gz, .tgz, .tar) will be extracted.
  --gdrive ID       Google Drive file id (requires 'gdown' Python package).
  --hf REPO_URL     Hugging Face model repo URL (git + git-lfs recommended). Example: https://huggingface.co/owner/repo
  --s3 S3_PATH      AWS S3 path (s3://bucket/path). Requires AWS CLI configured.
  --outdir DIR      Output directory to place the checkpoint files (will be created).
  -h, --help        Show this help

Examples:
  $0 --url https://example.com/checkpoint.zip --outdir /path/to/checkpoint_dir
  $0 --gdrive 1A2b3C4d5EFghIJ6 --outdir ./checkpoint_folder
  $0 --hf https://huggingface.co/owner/repo --outdir ./checkpoint_folder
  $0 --s3 s3://my-bucket/checkpoints/checkpoint.tar.gz --outdir ./checkpoint_folder

This script will try to extract common archive formats and print the final directory contents.
EOF
}

# parse args
URL=""
GDRIVE=""
HF=""
S3PATH=""
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      URL="$2"; shift 2;;
    --gdrive)
      GDRIVE="$2"; shift 2;;
    --hf)
      HF="$2"; shift 2;;
    --s3)
      S3PATH="$2"; shift 2;;
    --outdir)
      OUTDIR="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "$OUTDIR" ]]; then
  echo "Error: --outdir is required"
  usage
  exit 1
fi

mkdir -p "$OUTDIR"
WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR"

downloaded_file=""

if [[ -n "$URL" ]]; then
  echo "Downloading from URL: $URL"
  # Use curl if available, fallback to wget
  if command -v curl >/dev/null 2>&1; then
    curl -L "$URL" -o download.tmp
  elif command -v wget >/dev/null 2>&1; then
    wget -O download.tmp "$URL"
  else
    echo "Error: curl or wget required to download from URL"
    exit 1
  fi
  downloaded_file="download.tmp"
fi

if [[ -n "$GDRIVE" ]]; then
  echo "Downloading from Google Drive id: $GDRIVE"
  if ! command -v gdown >/dev/null 2>&1; then
    echo "gdown not found. Installing via pip..."
    python3 -m pip install --user gdown
    export PATH="$HOME/.local/bin:$PATH"
  fi
  gdown --id "$GDRIVE" -O download.tmp
  downloaded_file="download.tmp"
fi

if [[ -n "$HF" ]]; then
  echo "Cloning Hugging Face repo: $HF"
  if ! command -v git >/dev/null 2>&1; then
    echo "Error: git required to clone Hugging Face repos"
    exit 1
  fi
  # Try normal git clone; the user should have git-lfs installed if large files are present
  git clone "$HF" hf_repo
  echo "Copying repo contents to outdir..."
  rsync -a hf_repo/ "$OUTDIR/"
  echo "Done. Files copied to $OUTDIR"
  exit 0
fi

if [[ -n "$S3PATH" ]]; then
  echo "Downloading from S3: $S3PATH"
  if ! command -v aws >/dev/null 2>&1; then
    echo "Error: aws CLI not found. Install and configure AWS CLI to use S3 downloads."
    exit 1
  fi
  # If S3 path ends with slash, sync; otherwise copy file
  if [[ "$S3PATH" == */ ]]; then
    aws s3 sync "$S3PATH" "$OUTDIR/"
  else
    aws s3 cp "$S3PATH" "$WORKDIR/download.tmp"
    downloaded_file="download.tmp"
  fi
fi

# If we have a downloaded file, try to detect archive and extract
if [[ -n "$downloaded_file" ]]; then
  file -b --mime-type "$downloaded_file" || true
  mime=$(file -b --mime-type "$downloaded_file" || true)
  echo "Downloaded file mime-type: $mime"

  # try common archive extensions by filename first
  if [[ "$downloaded_file" == *.zip ]]; then
    echo "Extracting zip..."
    unzip -q "$downloaded_file" -d extracted
  elif [[ "$downloaded_file" == *.tar.gz ]] || [[ "$downloaded_file" == *.tgz ]]; then
    echo "Extracting tar.gz..."
    mkdir -p extracted
    tar -xzf "$downloaded_file" -C extracted
  elif [[ "$downloaded_file" == *.tar ]]; then
    echo "Extracting tar..."
    mkdir -p extracted
    tar -xf "$downloaded_file" -C extracted
  else
    # fallback to checking mime-type
    case "$mime" in
      application/zip)
        unzip -q "$downloaded_file" -d extracted;;
      application/x-gzip|application/gzip)
        mkdir -p extracted; tar -xzf "$downloaded_file" -C extracted;;
      application/x-tar)
        mkdir -p extracted; tar -xf "$downloaded_file" -C extracted;;
      *)
        echo "Unknown archive type or single file. Moving to outdir as-is."
        mv "$downloaded_file" "$OUTDIR/"
        echo "Done. File moved to $OUTDIR/"
        exit 0;
        ;;
    esac
  fi

  # If we extracted, move extracted contents into outdir
  if [[ -d extracted ]]; then
    echo "Moving extracted files to $OUTDIR"
    rsync -a extracted/ "$OUTDIR/"
    echo "Extraction complete. Listing $OUTDIR:"
    ls -la "$OUTDIR"
    exit 0
  fi
fi

# If we got here with no inputs processed, error
echo "No download performed. Provide --url, --gdrive, --hf or --s3"
usage
exit 1
