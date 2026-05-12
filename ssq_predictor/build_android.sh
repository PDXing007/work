#!/bin/bash
# ============================================================
# SSQ Predictor — Android APK 构建 + 签名脚本
#
# 前置条件 (Linux / WSL2 with sudo):
#   sudo apt update && sudo apt install -y python3-pip git zip unzip openjdk-17-jdk autoconf libtool libffi-dev libssl-dev
#   pip install --user buildozer cython
#
# 用法:
#   ./build_android.sh              # 构建 release APK (已签名)
#   ./build_android.sh --debug      # 构建 debug APK
# ============================================================

set -e

MODE="release"

for arg in "$@"; do
    case $arg in
        --debug) MODE="debug" ;;
    esac
done

echo "============================================"
echo " SSQ Predictor — APK Builder"
echo " Mode: $MODE"
echo "============================================"

if ! command -v buildozer &> /dev/null; then
    echo "[ERROR] buildozer not found"
    echo "  Install: pip install --user buildozer cython"
    echo "  Also: sudo apt install -y git zip unzip openjdk-17-jdk autoconf libtool libffi-dev libssl-dev"
    exit 1
fi

# Check keystore
if [ "$MODE" = "release" ] && [ ! -f "ssq-release.keystore" ]; then
    echo "[ERROR] ssq-release.keystore not found"
    exit 1
fi

echo ""
echo "[1/2] Building $MODE APK..."
echo "  First run downloads Android SDK/NDK (~1.5GB) — be patient"
echo ""

buildozer android $MODE 2>&1 | while read line; do
    case "$line" in
        *STEP*|*Download*|*Compile*|*Package*|*APK*|*Signed*|*ERROR*|*SUCCESS*)
            echo "  $line"
            ;;
    esac
done

echo ""
echo "============================================"
echo " BUILD COMPLETE"
echo "============================================"

if ls bin/*.apk 2>/dev/null; then
    echo ""
    echo "APK details:"
    ls -lh bin/*.apk
    echo ""
    if [ "$MODE" = "release" ]; then
        echo "Signed with: ssq-release.keystore (alias: ssqkey)"
        echo "Ready to install on Android devices!"
    fi
else
    echo "APK not found. Check .buildozer/ directory for logs."
    exit 1
fi
