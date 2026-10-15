#!/bin/sh
# macOS counterpart of launch_blender.ps1: starts Blender with dev_load.py.
# BLENDER_EXE may name the executable or the Blender.app bundle.

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

resolve_app_bundle() {
    if [ -d "$1" ] && [ "${1%.app}" != "$1" ]; then
        printf '%s/Contents/MacOS/Blender\n' "$1"
    else
        printf '%s\n' "$1"
    fi
}

find_blender() {
    if [ -n "${BLENDER_EXE:-}" ]; then
        resolve_app_bundle "$BLENDER_EXE"
        return 0
    fi

    if command -v blender >/dev/null 2>&1; then
        command -v blender
        return 0
    fi

    # Glob results are sorted, so the last match is the newest version.
    found=""
    for candidate in \
        /Applications/Blender*.app/Contents/MacOS/Blender \
        "$HOME"/Applications/Blender*.app/Contents/MacOS/Blender; do
        if [ -x "$candidate" ]; then
            found=$candidate
        fi
    done

    if [ -n "$found" ]; then
        printf '%s\n' "$found"
        return 0
    fi

    return 1
}

if ! blender_exe=$(find_blender) || [ ! -x "$blender_exe" ]; then
    echo "Blender was not found. Install Blender in /Applications, add blender to PATH, or set BLENDER_EXE to the Blender executable or Blender.app." >&2
    exit 1
fi

echo "Starting $blender_exe"
exec "$blender_exe" --python "$script_dir/dev_load.py"
