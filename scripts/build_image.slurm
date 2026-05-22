#!/bin/bash
#SBATCH --job-name=build_woodland
#SBATCH --partition=defq              # İşlemci odaklı (CPU) kuyruk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8              # Build işlemini hızlandırmak için 8 core
#SBATCH --time=01:00:00                # Maksimum 1 saat süre
#SBATCH --mem=32G                      # Katmanları açmak için geniş bellek
#SBATCH --output=slurm_logs/build_%j.out
#SBATCH --error=slurm_logs/build_%j.err


CONTAINER_IMAGE="$HOME/containers/woodland-ext.sif"
TAR_ARCHIVE="$HOME/woodland/woodland-ext_latest.tar.gz"

echo "SIF build işlemi başladı..."

# Eğer eski bir SIF varsa temizle
if [ -f "$CONTAINER_IMAGE" ]; then
    rm "$CONTAINER_IMAGE"
    echo "Eski SIF silindi."
fi

# Build komutu
apptainer build "$CONTAINER_IMAGE" docker-archive://"$TAR_ARCHIVE"

if [ $? -eq 0 ]; then
    echo "$CONTAINER_IMAGE oluşturuldu."
    rm -f "$TAR_ARCHIVE"
    echo "tar.gz silindi: $TAR_ARCHIVE"
else
    echo "HATA: Build başarısız. tar.gz korundu."
    exit 1
fi
